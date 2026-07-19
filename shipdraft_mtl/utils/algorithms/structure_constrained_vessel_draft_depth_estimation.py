"""
船舶吃水深度估计模块 (Ship Draft Depth Estimator) - Global Grid Matching Edition v7.2

基于全局网格重建与模板匹配技术 (Global Grid Reconstruction & Template Matching)

核心改进 (v7.2):
    1. 优化代价函数: 提高步长一致性权重 λ_cons = 2.0，收紧判定区间
       - J(α) = J_phys + λ_cons × J_cons + J_sem
       - J_cons: [0.95, 1.05] → +4.0, [0.85, 1.15] → +2.0, 其他 → -2.0
    2. 改进深度回归: 单点回归使用局部最优步长（靠近水线的分段加权平均）
       - D = V_0 + k_theory × (t_wl - ρ_0), 其中 k_theory = -0.2 / S_opt

核心改进 (v7):
    1. 最优因子搜索: 移除固定的 ADAPTIVE_FACTOR = 1.8，改为在 [1.4, 2.2] 范围内搜索最优因子
    2. 多维度评估: 基于Structure-Constrained、步长一致性和字符匹配三个维度评估最优行数
    3. 自适应分段: 每个分段独立计算最优因子，更好地适应透视畸变

核心改进 (v4):
    1. 统一向量计算: 移除 is_vertical 分支，垂直情况只是倾斜角度=0的特例
    2. 消除重复投影: 检测框只投影一次，全程复用投影结果
    3. 简化水线计算: 统一流程 - 获取刻度X范围 → 每列mask最小Y → 中位数Y → 主轴交点

算法流程:
    1. 预处理: 跨类别 NMS 去重
    2. 倾斜检测: 通过检测框中心点拟合主轴方向（垂直=0度倾斜）
    3. 一次性投影: 所有检测框中心投影到主轴，缓存结果
    4. 网格构建: 
       - 聚类检测框为物理行组
       - 对每个分段搜索最优步长因子 (1.4-2.2)
       - 按投影坐标分行，构建N行的"盒子列表"
    5. 模板匹配:
       - 生成标准刻度模板 (6.0m - 0.2m, 间隔0.2m)
       - 滑动窗口进行行内集合匹配
    6. 水线计算:
       - 获取刻度X范围 [Xmin, Xmax]
       - 在该范围内每列取mask=1的最小Y
       - 计算这些Y的中位数 Y_wl
       - 计算水平线 y=Y_wl 与主轴的交点坐标
    7. 深度回归:
       - 线性回归拟合 (投影坐标 -> Depth M)
       - 代入水线投影坐标计算深度
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

# =============================================================================
# 常量定义
# =============================================================================

SCALE_INTERVAL = 0.2        # 相邻刻度物理间隔 (米)
TEMPLATE_MAX_DEPTH = 6.0    # 模板最大深度 (米)
TEMPLATE_MIN_DEPTH = 0.2    # 模板最小深度 (米)
NMS_IOU_THRESHOLD = 0.3     # NMS 阈值

# 自适应步长因子搜索范围 (V7)
# 物理含义: 相邻刻度行间距 ≈ 检测框高度 × 因子
# 典型范围: 1.4 (紧凑) ~ 2.2 (稀疏)
FACTOR_SEARCH_MIN = 1.4     # 最小步长因子
FACTOR_SEARCH_MAX = 2.0     # 最大步长因子
FACTOR_SEARCH_STEP = 0.01   # 搜索步长

# 水线提取预处理参数 (V7.1)
# 用于过滤分割掩码中的离群小区域，提高水线提取精度
MASK_PREPROCESS_ENABLED = True          # 是否启用 mask 预处理（可选，用于对比实验）
MASK_MORPH_KERNEL_SIZE = 5              # 形态学操作核大小
MASK_MIN_AREA_RATIO = 0.01              # 最小连通区域面积比例（相对于图像面积）

# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class DetBox:
    """内部使用的检测框结构"""
    cls_name: str
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    
    # 【v4优化】缓存计算结果，避免重复投影
    # _proj_main: 在主轴上的投影坐标 (用于深度计算)
    # _proj_normal: 在法线上的投影坐标 (用于行内排序)
    _proj_main: Optional[float] = field(default=None, repr=False)
    _proj_normal: Optional[float] = field(default=None, repr=False)

    @property
    def x_center(self) -> float: return (self.x1 + self.x2) / 2
    @property
    def y_center(self) -> float: return (self.y1 + self.y2) / 2
    @property
    def height(self) -> float: return self.y2 - self.y1
    @property
    def width(self) -> float: return self.x2 - self.x1

@dataclass
class GridRow:
    """网格行 - 柔性行容器"""
    row_idx: int
    boxes: List[DetBox] = field(default_factory=list)
    matched_val: Optional[float] = None
    proj_coord: float = 0.0  # 该行在主轴上的理论投影坐标

@dataclass
class ScaleReading:
    """最终输出的刻度读数"""
    value: float          # 真实物理值 (校正后)
    y_center: float       # 像素坐标 (原始Y坐标，用于可视化)
    is_inferred: bool     # 是否是推断/补全的
    raw_char: Optional[str] = None
    proj_coord: Optional[float] = None  # 主轴投影坐标

@dataclass
class DepthEstimationResult:
    """深度估计结果详单"""
    depth: Optional[float]
    waterline_y: float
    waterline_xy: Optional[Tuple[float, float]]  # 水线与主轴交点坐标
    scales: List[ScaleReading]
    method: str
    success: bool
    debug_info: Dict = field(default_factory=dict)

@dataclass 
class AxisInfo:
    """
    主轴信息 - 统一向量表示，无需区分垂直/倾斜
    
    垂直情况只是 angle=0, direction=[0,1] 的特例
    """
    angle: float          # 倾斜角度 (弧度)，正值表示顺时针
    direction: np.ndarray # 主轴方向单位向量 [dx, dy]，向下为正
    normal: np.ndarray    # 垂直于主轴的方向 [nx, ny]，向右为正
    origin: np.ndarray    # 参考原点 [x0, y0]

# =============================================================================
# 核心算法：网格匹配器
# =============================================================================

class GlobalTemplate:
    """标准物理刻度模板"""
    def __init__(self):
        self.sequence = []
        # 生成 6.0 到 0.2 的序列
        start = int(TEMPLATE_MAX_DEPTH * 10)
        end = int(TEMPLATE_MIN_DEPTH * 10)
        
        for i in range(start, end - 2, -2):
            val = round(i / 10.0, 1)
            int_part = int(val)
            dec_part = int(round((val - int_part) * 10))
            
            str_int = str(int_part)
            str_dec = 'M' if dec_part == 0 else str(dec_part)
            
            self.sequence.append({
                'int_char': str_int,
                'dec_char': str_dec,
                'value': val
            })

class GridPatternMatcher:
    """
    网格模式匹配器 - 统一向量版本
    """
    def __init__(self):
        self.template = GlobalTemplate()
        self.axis_info: Optional[AxisInfo] = None

    def process(self, boxes: List[DetBox]) -> Tuple[List[ScaleReading], Dict]:
        """处理检测框列表，返回重构后的刻度"""
        debug_info = {}
        
        # 1. NMS 去重
        boxes = self._nms(boxes)
        if not boxes:
            return [], {"error": "No boxes after NMS"}

        # 2. 检测主轴方向（统一向量表示）
        self.axis_info = self._detect_axis(boxes)
        debug_info['axis_angle_deg'] = np.degrees(self.axis_info.angle)

        # 3. 【v4优化】一次性投影所有检测框，缓存结果
        self._project_all_boxes(boxes)

        # 4. 构建网格矩阵 (使用缓存投影)
        grid_matrix, row_step, min_proj = self._build_grid_matrix(boxes)
        if not grid_matrix:
            return [], {"error": "Failed to build matrix"}
        
        debug_info['row_step'] = row_step
        debug_info['matrix_rows'] = len(grid_matrix)

        # 5. 全局模板匹配
        best_start_idx, best_score = self._sliding_window_match(grid_matrix)
        debug_info['best_match_score'] = best_score

        # 6. 数据重构
        if best_score < -10: 
            return [], {"error": "Low matching score"}
             
        final_scales = self._reconstruct_results(grid_matrix, best_start_idx, row_step, min_proj)
        
        return final_scales, debug_info

    def _detect_axis(self, boxes: List[DetBox]) -> AxisInfo:
        """
        检测主轴方向 - 统一向量表示
        
        垂直情况：angle=0, direction=[0, 1]
        倾斜情况：通过PCA检测，direction=[sin(θ), cos(θ)]
        """
        # 收集所有检测框中心点
        centers = np.array([[b.x_center, b.y_center] for b in boxes])
        
        # 找到Y最小的检测框（最上方）作为默认原点
        top_idx = np.argmin(centers[:, 1])
        origin = centers[top_idx].copy()
        
        # 默认垂直方向
        default_axis = AxisInfo(
            angle=0.0,
            direction=np.array([0.0, 1.0]),  # 垂直向下
            normal=np.array([1.0, 0.0]),     # 水平向右
            origin=origin
        )
        
        # 样本太少或分布太集中，回退到默认
        if len(boxes) < 3: return default_axis
        
        avg_h = np.mean([b.height for b in boxes])
        avg_w = np.mean([b.width for b in boxes])
        
        x_span = np.max(centers[:, 0]) - np.min(centers[:, 0])
        y_span = np.max(centers[:, 1]) - np.min(centers[:, 1])
        
        # 关键几何先验检查
        if y_span < avg_h * 0.8: return default_axis # 水平分布
        if x_span < avg_w * 0.8: return default_axis # 垂直分布
        
        # PCA 主成分分析
        centered = centers - centers.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # 第一主成分
        principal_dir = eigenvectors[:, -1]
        
        # 确保方向向下
        if principal_dir[1] < 0:
            principal_dir = -principal_dir
        
        # 计算角度
        angle = np.arctan2(principal_dir[0], principal_dir[1])
        
        # 限制最大倾斜角度 (20度)
        if abs(np.degrees(angle)) > 20:
            return default_axis
        
        # 归一化方向向量
        direction = principal_dir / np.linalg.norm(principal_dir)
        # 【修复】normal 向右 [dy, -dx]，确保 X 大的点 _proj_normal 大
        # 这样排序后: 左边(整数位) -> 右边(小数位)
        normal = np.array([direction[1], -direction[0]])
        
        # 对于倾斜情况，使用所有点在主轴上的投影最小值点作为原点
        # 这样原点更稳定，不会因为最上方点左右抖动而变化
        # proj = dot(C - mean, dir)
        mean_center = centers.mean(axis=0)
        proj_vals = [np.dot(c - mean_center, direction) for c in centers]
        min_proj_idx = np.argmin(proj_vals)
        origin = centers[min_proj_idx].copy()
        
        return AxisInfo(
            angle=angle,
            direction=direction,
            normal=normal,
            origin=origin
        )

    def _project_all_boxes(self, boxes: List[DetBox]) -> None:
        """
        【v4优化】一次性计算所有检测框的投影，缓存到 DetBox 对象中
        """
        if self.axis_info is None: return
        
        origin = self.axis_info.origin
        dx, dy = self.axis_info.direction
        nx, ny = self.axis_info.normal
        
        for box in boxes:
            rel_x = box.x_center - origin[0]
            rel_y = box.y_center - origin[1]
            
            # 向量点积投影
            box._proj_main = float(rel_x * dx + rel_y * dy)
            box._proj_normal = float(rel_x * nx + rel_y * ny)

    def compute_waterline_intersection(self, waterline_y: float) -> Tuple[float, Tuple[float, float]]:
        """
        计算水平水线与主轴的交点，返回投影坐标和交点坐标
        
        几何原理（统一公式，无分支）：
        - 水线方程: y = Y_wl
        - 主轴方程: P = P0 + t * direction
        - 交点: t = (Y_wl - Y0) / dy
        - 投影坐标: proj = t (因为direction是单位向量)
        """
        if self.axis_info is None:
            return waterline_y, (0.0, waterline_y)
        
        x0, y0 = self.axis_info.origin
        dx, dy = self.axis_info.direction
        
        # 避免除零（dy=0 意味着主轴水平，异常）
        if abs(dy) < 1e-6:
            return waterline_y - y0, (x0, waterline_y)
        
        # 计算交点参数 t (即投影距离)
        t = (waterline_y - y0) / dy
        
        # 交点坐标 (用于可视化)
        x_cross = x0 + t * dx
        y_cross = waterline_y
        
        return t, (x_cross, y_cross)

    def inverse_project_to_y(self, proj_coord: float) -> float:
        """从投影坐标反推原始Y坐标: P = P0 + proj * dir"""
        if self.axis_info is None: return proj_coord
        
        # y = y0 + proj * dy
        return float(self.axis_info.origin[1] + proj_coord * self.axis_info.direction[1])

    def _nms(self, boxes: List[DetBox], iou_thresh=0.3) -> List[DetBox]:
        if not boxes: return []
        boxes = sorted(boxes, key=lambda x: x.conf, reverse=True)
        keep = []
        while boxes:
            curr = boxes.pop(0)
            keep.append(curr)
            boxes = [b for b in boxes if self._iou(curr, b) < iou_thresh]
        return keep

    def _iou(self, a: DetBox, b: DetBox) -> float:
        xx1 = max(a.x1, b.x1); yy1 = max(a.y1, b.y1)
        xx2 = min(a.x2, b.x2); yy2 = min(a.y2, b.y2)
        w = max(0, xx2 - xx1); h = max(0, yy2 - yy1)
        inter = w * h
        return inter / (a.width*a.height + b.width*b.height - inter + 1e-6)

    # =========================================================================
    # V7 核心算法：自适应分段网格构建 + 最优因子搜索 (Adaptive Segmented Grid with Optimal Factor Search)
    # =========================================================================

    def _cluster_boxes_by_projection(self, boxes: List[DetBox]) -> List[List[DetBox]]:
        """
        步骤1: 将投影距离非常近的框聚类为"物理行组"
        """
        if not boxes: return []
        
        # 按主轴投影排序
        sorted_boxes = sorted(boxes, key=lambda b: b._proj_main)
        
        clusters = []
        current_cluster = [sorted_boxes[0]]
        
        # 聚类阈值：平均框高的 0.5 倍 (如果两个框投影差距小于半个字高，认为是同一行)
        avg_h = np.mean([b.height for b in boxes])
        merge_thresh = avg_h * 0.5
        
        for i in range(1, len(sorted_boxes)):
            box = sorted_boxes[i]
            prev_box = current_cluster[-1]
            
            # 比较当前框与簇内最后一个框的投影距离
            dist = box._proj_main - prev_box._proj_main
            
            if dist < merge_thresh:
                current_cluster.append(box)
            else:
                clusters.append(current_cluster)
                current_cluster = [box]
        
        clusters.append(current_cluster)
        return clusters

    def _search_optimal_factor(self, curr_cluster: List[DetBox], next_cluster: List[DetBox],
                                local_h: float, dist: float,
                                prev_pixel_step: Optional[float] = None) -> Tuple[int, float, float]:
        """
        【V7.3优化】搜索最优步长因子 - 新增前后分段步长一致性评分
        
        在 [FACTOR_SEARCH_MIN, FACTOR_SEARCH_MAX] 范围内搜索最优的步长因子，
        使得推断的行数最合理。
        
        代价函数: J(α) = λ_cons × J_cons + J_sem + λ_seq × J_seq
        其中 λ_cons = 2.0, λ_seq = 2.5
        
        评估标准：
        1. 步长一致性 J_cons：实际步长与估计步长的偏差越小越好
        2. 语义匹配 J_sem：检测框字符与模板的匹配程度
        3. 前后分段一致性 J_seq (V7.3新增)：当前分段步长与上一分段步长的比值
           应接近1.0（透视畸变是相邻递进收缩的，不会突变）
        
        Args:
            curr_cluster: 当前簇
            next_cluster: 下一个簇
            local_h: 局部平均框高
            dist: 两簇之间的投影距离
            prev_pixel_step: 上一分段的最优像素步长（首段为None）
            
        Returns:
            (最优行数, 最优因子, 最优像素步长)
        """
        # 使用全局常量定义的搜索范围
        best_delta_rows = 1
        best_factor = (FACTOR_SEARCH_MIN + FACTOR_SEARCH_MAX) / 2  # 默认中间值
        best_score = -float('inf')
        best_pixel_step = local_h * best_factor  # 最优像素步长
        
        # 权重定义
        LAMBDA_CONS = 2.0   # 步长一致性权重
        LAMBDA_SEQ = 1.5    # 前后分段一致性权重 (V7.3)
        
        # 获取簇内字符用于匹配评估
        curr_chars = set(b.cls_name for b in curr_cluster)
        next_chars = set(b.cls_name for b in next_cluster)
        
        # 整数位字符集合
        int_chars = {'0', '1', '2', '3', '4', '5', '6'}
        curr_ints = curr_chars & int_chars
        next_ints = next_chars & int_chars
        
        # 遍历所有可能的因子
        factor = FACTOR_SEARCH_MIN
        while factor <= FACTOR_SEARCH_MAX + 1e-6:
            estimated_step = local_h * factor
            
            # 计算对应的行数
            if estimated_step > 0:
                delta_rows = max(1, int(round(dist / estimated_step)))
            else:
                delta_rows = 1
            
            # =====================================================
            # 评估1: 步长一致性 J_cons
            # r = (D_proj / ΔR) / (h_local × α)
            # =====================================================
            actual_step = dist / delta_rows if delta_rows > 0 else dist
            step_ratio = actual_step / estimated_step if estimated_step > 0 else 1.0
            
            J_cons = 0.0
            if 0.97 <= step_ratio <= 1.03:
                J_cons = 4.0   # 极高吻合
            elif 0.85 <= step_ratio <= 1.15:
                J_cons = 2.0   # 较好吻合
            elif 0.7 <= step_ratio <= 1.3:
                J_cons = 0.0   # 可接受
            else:
                J_cons = -2.0  # 严重偏差
            
            # =====================================================
            # 评估2: 语义匹配约束 J_sem
            # =====================================================
            J_sem = 0.0
            if curr_ints and next_ints:
                if curr_ints == next_ints:
                    # 整数位相同，期望较小间隔
                    if 1 <= delta_rows <= 4:
                        J_sem = 2.0
                    elif delta_rows > 5:
                        J_sem = -1.0
            
            # =====================================================
            # 评估3: 前后分段步长一致性 J_seq (V7.3新增)
            # R = S_curr / S_prev，透视畸变导致步长递进缩放，不应突变
            # =====================================================
            J_seq = 0.0
            if prev_pixel_step is not None and prev_pixel_step > 0:
                R = actual_step / prev_pixel_step
                if 0.95 <= R <= 1.05:
                    J_seq = 4.0    # 步长极度稳定，大概率正确
                elif 0.80 <= R <= 1.20:
                    J_seq = 2.0    # 步长轻微缩放，符合透视递进
                elif 0.60 <= R <= 1.40:
                    J_seq = 0.0    # 可接受范围
                else:
                    J_seq = -4.0   # 步长突变，可能多算或少算一行
            
            # =====================================================
            # 总代价: J = λ_cons × J_cons + J_sem + λ_seq × J_seq
            # =====================================================
            score = LAMBDA_CONS * J_cons + J_sem + LAMBDA_SEQ * J_seq
            
            # 更新最优解
            if score > best_score:
                best_score = score
                best_delta_rows = delta_rows
                best_factor = factor
                best_pixel_step = actual_step  # 记录实际像素步长
            
            factor += FACTOR_SEARCH_STEP
        
        return best_delta_rows, best_factor, best_pixel_step

    def _build_grid_matrix(self, boxes: List[DetBox]) -> Tuple[List[GridRow], float, float]:
        """
        【V7.2重构】自适应分段网格构建 + 最优因子搜索
        
        改进点：
        1. 不再使用固定的 ADAPTIVE_FACTOR
        2. 对每个分段独立搜索最优因子（范围由 FACTOR_SEARCH_MIN/MAX 定义）
        3. 基于Structure-Constrained、步长一致性和字符匹配评估最优行数
        4. 【V7.2】返回局部最优步长用于单点回归
        """
        if not boxes: return [], 0, 0

        # 1. 聚类：将框分组 (Visual Rows)
        clusters = self._cluster_boxes_by_projection(boxes)
        
        # 2. 初始化网格行容器
        grid_map: Dict[int, List[DetBox]] = {}
        
        # 第一个簇默认为第0行
        current_row_idx = 0
        grid_map[0] = clusters[0]
        
        # 记录每行的理论投影坐标 (用于填补空行)
        row_proj_coords = {0: np.mean([b._proj_main for b in clusters[0]])}
        
        # 记录平均步长和因子用于Debug
        step_history = []
        factor_history = []
        optimal_pixel_steps = []  # 【V7.2】记录每段的最优像素步长
        prev_pixel_step = None    # 【V7.3】上一分段的最优像素步长
        
        # 3. 链式构建（带最优因子搜索）
        for i in range(len(clusters) - 1):
            curr_c = clusters[i]
            next_c = clusters[i+1]
            
            # 计算当前位置的局部参考高度 (Local Height)
            h_curr = np.mean([b.height for b in curr_c])
            h_next = np.mean([b.height for b in next_c])
            local_h = (h_curr + h_next) / 2.0
            
            # 计算簇中心投影距离
            proj_curr = np.mean([b._proj_main for b in curr_c])
            proj_next = np.mean([b._proj_main for b in next_c])
            dist = proj_next - proj_curr
            
            # 【V7.3核心】搜索最优因子，传入上一分段步长用于一致性评估
            delta_rows, best_factor, best_pixel_step = self._search_optimal_factor(
                curr_c, next_c, local_h, dist, prev_pixel_step
            )
            
            # 记录这一段的实际平均步长和使用的因子
            actual_segment_step = dist / delta_rows
            step_history.extend([actual_segment_step] * delta_rows)
            factor_history.append(best_factor)
            optimal_pixel_steps.append(best_pixel_step)  # 【V7.2】
            prev_pixel_step = best_pixel_step  # 【V7.3】更新上一分段步长
            
            # 更新行号
            current_row_idx += delta_rows
            grid_map[current_row_idx] = next_c
            row_proj_coords[current_row_idx] = proj_next
            
            # 填补中间的空行 (Ghost Rows)
            for k in range(1, delta_rows):
                ghost_idx = current_row_idx - delta_rows + k
                ghost_proj = proj_curr + k * actual_segment_step
                row_proj_coords[ghost_idx] = ghost_proj

        # 4. 转换为 List[GridRow]
        if not grid_map: return [], 0, 0
        
        max_idx = max(grid_map.keys())
        grid_rows = []
        
        # 【V7.2】计算步长
        # - 多刻度: 使用靠近水线的分段加权平均步长
        # - 单刻度: 使用经验公式 avg_box_height × 1.8
        if optimal_pixel_steps:
            # 多刻度情况：取最后 min(3, len) 段的平均值（更接近水线）
            recent_steps = optimal_pixel_steps[-min(3, len(optimal_pixel_steps)):]
            avg_step = float(np.mean(recent_steps))
        else:
            # 单刻度情况：无分段信息，使用经验公式
            # 步长 = 平均检测框高度 × 1.8
            avg_box_height = np.mean([b.height for b in boxes])
            avg_step = float(avg_box_height * 1.8)
        
        for r in range(max_idx + 1):
            boxes_in_row = grid_map.get(r, [])
            proj = row_proj_coords.get(r, 0.0)
            
            if boxes_in_row:
                boxes_in_row.sort(key=lambda x: x._proj_normal)
                
            grid_rows.append(GridRow(r, boxes_in_row, proj_coord=proj))
            
        return grid_rows, avg_step, 0.0

    def _sliding_window_match(self, grid_rows: List[GridRow]) -> Tuple[int, float]:
        """
        基于内容的行匹配 (v5改进: 引入置信度加权)
        
        Score = Sum( Weight * (Match ? 1 : -0.5) )
        Weight = Box_Confidence
        """
        template_seq = self.template.sequence
        mat_rows = len(grid_rows)
        tmpl_len = len(template_seq)
        
        best_score = -float('inf')
        best_start_idx = 0
        
        # 如果矩阵行数比模板还长，说明出错了或包含了太多噪声，截断处理
        search_range = tmpl_len - mat_rows + 1
        
        # 特殊情况处理
        if search_range <= 0:
            # 尝试匹配重叠部分最大的区域，或者返回错误
            # 这里简单处理：只搜索能匹配的区域
            search_range = 1
        
        # 基础分值定义
        SCORE_MATCH = 10.0
        SCORE_MISMATCH = -10.0  # 加大惩罚
        SCORE_PARTIAL = 5.0     # 模糊匹配 (如 3 识别成 8)
        
        for i in range(search_range):
            current_score = 0.0
            valid_box_count = 0
            
            for r in range(mat_rows):
                row_boxes = grid_rows[r].boxes
                
                # 跳过空行，空行不加分也不减分（允许缺失）
                if not row_boxes: 
                    continue
                
                # 防止索引越界
                if i + r >= len(template_seq): break
                
                tmpl_item = template_seq[i + r]
                target_int = tmpl_item['int_char']
                target_dec = tmpl_item['dec_char']
                
                left_box, right_box = self._classify_left_right(row_boxes)
                
                # 辅助函数：计算单框得分
                def get_box_score(box, target_char):
                    if box is None: return 0.0
                    
                    # 基础分值 * 置信度
                    w = box.conf
                    if box.cls_name == target_char:
                        return SCORE_MATCH * w
                    else:
                        # 字符混淆矩阵 (简单的硬编码示例)
                        # 3和8, 5和6, 1和7 容易混淆
                        is_confusing = (box.cls_name in '38' and target_char in '38') or \
                                     (box.cls_name in '56' and target_char in '56')
                        if is_confusing:
                            return SCORE_PARTIAL * w
                        else:
                            return SCORE_MISMATCH * w

                # 计算左右两列的得分
                s_int = get_box_score(left_box, target_int)
                s_dec = get_box_score(right_box, target_dec)
                
                current_score += (s_int + s_dec)
                if left_box: valid_box_count += 1
                if right_box: valid_box_count += 1
            
            # 归一化分数 (可选，防止因框多分数虚高，但这里累加和更好，因为匹配越多越好)
            # 只要总分高，说明置信度高且匹配准确的数量多
            
            if current_score > best_score:
                best_score = current_score
                best_start_idx = i
        
        # 如果最高分太低（比如全是负分），说明匹配失败
        if best_score < -5: 
            # 降级策略：如果全局匹配失败，尝试寻找单个最高置信度的锚点 (TODO)
            pass
            
        return best_start_idx, best_score


    def _classify_left_right(self, row_boxes: List[DetBox]) -> Tuple[Optional[DetBox], Optional[DetBox]]:
        """
        【v4.1新增】基于法线投影 _proj_normal 明确区分左右列框
        
        原理：
        - _proj_normal < 0: 在主轴左侧（整数位）
        - _proj_normal > 0: 在主轴右侧（小数位）
        - _proj_normal ≈ 0: 在主轴上（需要特殊处理）
        
        这种方法在倾斜场景下比简单使用X坐标更准确
        
        Returns:
            (left_box, right_box): 左列框（整数位）和右列框（小数位），可能为None
        """
        if not row_boxes:
            return None, None
        
        if len(row_boxes) == 1:
            # 单框情况：根据 _proj_normal 判断是左还是右
            box = row_boxes[0]
            if box._proj_normal is not None:
                if box._proj_normal < 0:
                    return box, None  # 左列（整数位）
                else:
                    return None, box  # 右列（小数位）
            return box, None  # 默认当作左列
        
        # 多框情况：找出最左和最右的框
        # 由于已经按 _proj_normal 排序，第一个是最左，最后一个是最右
        left_box = row_boxes[0]   # _proj_normal 最小
        right_box = row_boxes[-1]  # _proj_normal 最大
        
        # 额外验证：确保左框确实在左侧（_proj_normal 较小）
        # 如果两个框的 _proj_normal 差异很小，可能是同一列的误检
        if left_box._proj_normal is not None and right_box._proj_normal is not None:
            proj_diff = right_box._proj_normal - left_box._proj_normal
            avg_width = (left_box.width + right_box.width) / 2
            
            # 如果投影差异小于平均宽度的0.5倍，认为是同一列
            if proj_diff < avg_width * 0.5:
                # 同一列，根据平均 _proj_normal 判断是左还是右
                avg_proj = (left_box._proj_normal + right_box._proj_normal) / 2
                if avg_proj < 0:
                    return left_box, None  # 都在左列
                else:
                    return None, right_box  # 都在右列
        
        return left_box, right_box

    

    def _reconstruct_results(self, grid_rows: List[GridRow], start_idx: int, 
                            row_step: float, min_proj: float) -> List[ScaleReading]:
        """重构结果"""
        final_results = []
        tmpl = self.template.sequence
        
        for r in range(len(grid_rows)):
            if start_idx + r >= len(tmpl): break
            val = tmpl[start_idx + r]['value']
            row_boxes = grid_rows[r].boxes
            proj_coord = grid_rows[r].proj_coord
            
            if row_boxes:
                # 使用真实检测框的均值
                avg_proj = np.mean([b._proj_main for b in row_boxes])
                avg_y = np.mean([b.y_center for b in row_boxes])
                raw_char = ",".join([b.cls_name for b in row_boxes])
                is_inf = False
            else:
                # 推断
                avg_proj = proj_coord
                avg_y = self.inverse_project_to_y(avg_proj)
                raw_char = None
                is_inf = True
                
            final_results.append(ScaleReading(val, float(avg_y), is_inf, raw_char, float(avg_proj)))
            
        return final_results
    
    def get_scale_x_range(self, boxes: List[DetBox]) -> Tuple[float, float]:
        """获取刻度区域的X范围"""
        if not boxes: return 0, 0
        xs = [b.x_center for b in boxes]
        avg_w = np.mean([b.width for b in boxes])
        return min(xs) - avg_w, max(xs) + avg_w

# =============================================================================
# 主估计器类
# =============================================================================

class StructureConstrainedVesselDraftDepthEstimation:
    DEFAULT_CLASS_NAMES = ['0', '1', '2', '3', '4', '6', '8', 'M', '5']

    def __init__(self, class_names=None, water_class_id=1, min_confidence=0.05,
                 mask_preprocess=None):
        """
        初始化深度估计器
        
        Args:
            class_names: 类别名称列表
            water_class_id: 水体类别ID
            min_confidence: 最小置信度阈值
            mask_preprocess: 是否启用 mask 预处理（过滤离群小区域）
                            None: 使用全局常量 MASK_PREPROCESS_ENABLED
                            True/False: 强制启用/禁用
        """
        self.class_names = class_names or self.DEFAULT_CLASS_NAMES
        self.water_class_id = water_class_id
        self.min_confidence = min_confidence
        self.matcher = GridPatternMatcher()
        
        # mask 预处理开关
        self.mask_preprocess = mask_preprocess if mask_preprocess is not None else MASK_PREPROCESS_ENABLED

    def estimate(self, det_boxes, det_labels, det_scores, seg_mask):
        res = self.estimate_with_details(det_boxes, det_labels, det_scores, seg_mask)
        return res.depth

    def estimate_with_details(self, det_boxes, det_labels, det_scores, seg_mask):
        # 1. 提取框
        raw_boxes = self._extract_boxes(det_boxes, det_labels, det_scores)
        
        # 2. 如果没有框，直接尝试提水线返回失败
        if not raw_boxes:
            wy = self._extract_waterline_simple(seg_mask, None)
            return self._fail_result(wy, "No boxes")

        # 3. 核心算法: 匹配与重构
        scales, debug = self.matcher.process(raw_boxes)
        
        # 4. 获取刻度X范围
        x_range = self.matcher.get_scale_x_range(raw_boxes)
        debug['scale_x_range'] = x_range
        
        # 5. 【v4优化】统一的水线提取流程
        # X范围 -> 列最小值 -> 中位数
        waterline_y = self._extract_waterline_simple(seg_mask, x_range)
        debug['waterline_y_raw'] = waterline_y
        
        # 【v7.3修复】补偿字符中心与物理刻度线的系统性偏差
        # 物理刻度线位于字符底边，而非中心；水线上移 0.3 个字符高度以修正
        if waterline_y > 0 and raw_boxes:
            avg_box_h = np.mean([b.height for b in raw_boxes])
            waterline_y = waterline_y - (0.3 * avg_box_h)
            debug['waterline_y_compensated'] = waterline_y
            debug['waterline_compensation_px'] = 0.3 * avg_box_h
        
        if not scales:
            return self._fail_result(waterline_y, "Matching failed", debug)

        # 6. 过滤倒影 (Structure-Constrained: 刻度在水线之上, Y更小)
        if waterline_y > 0:
            valid_scales = [s for s in scales if s.y_center < waterline_y]
        else:
            valid_scales = scales
            
        if not valid_scales:
            return self._fail_result(waterline_y, "All scales filtered", debug)

        # 7. 计算水线与主轴交点及投影
        wl_proj, wl_xy = self.matcher.compute_waterline_intersection(waterline_y)
        debug['waterline_proj'] = wl_proj
        debug['waterline_xy'] = wl_xy
        
        # 8. 深度回归 (使用投影坐标)
        depth = self._regress_depth(valid_scales, wl_proj, debug.get('row_step'))
        
        return DepthEstimationResult(depth, waterline_y, wl_xy, valid_scales, "GridV4", depth is not None, debug)

    def _extract_boxes(self, boxes, labels, scores):
        res = []
        if len(boxes) == 0: return res
        boxes = np.array(boxes).reshape(-1, 4)
        labels = np.array(labels).flatten()
        scores = np.array(scores).flatten()
        for b, l, s in zip(boxes, labels, scores):
            if s < self.min_confidence: continue
            l = int(l)
            if l >= len(self.class_names): continue
            name = self.class_names[l]
            if name == 'water': continue
            res.append(DetBox(name, float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(s)))
        return res

    def _clean_water_mask(self, water_mask: np.ndarray) -> np.ndarray:
        """
        【V7.1新增】清理水体掩码，过滤离群小区域
        
        水体在物理上是单一连通的，分割结果中的小离群区域通常是噪声。
        使用形态学开运算和连通区域分析，只保留主要的水体区域。
        
        Args:
            water_mask: (H, W) 二值水体掩码 (uint8, 0/1)
            
        Returns:
            清理后的水体掩码
        """
        H, W = water_mask.shape
        cleaned_mask = water_mask.copy()
        
        # 步骤1: 形态学开运算去除小噪点
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (MASK_MORPH_KERNEL_SIZE, MASK_MORPH_KERNEL_SIZE)
        )
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
        
        # 步骤2: 连通区域分析，只保留足够大的区域
        contours, _ = cv2.findContours(
            cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            # 如果形态学操作后没有轮廓，返回原始掩码
            return water_mask
        
        # 计算最小面积阈值
        total_area = H * W
        min_area = total_area * MASK_MIN_AREA_RATIO
        
        # 创建新的掩码，只包含足够大的连通区域
        filtered_mask = np.zeros_like(cleaned_mask)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                cv2.fillPoly(filtered_mask, [contour], 1)
        
        # 如果过滤后没有有效区域，保留最大的连通区域
        if filtered_mask.sum() == 0 and len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.fillPoly(filtered_mask, [largest_contour], 1)
        
        return filtered_mask

    def _extract_waterline_simple(self, seg_mask, x_range):
        """
        统一的水线提取逻辑
        
        【V7.1改进】可选的 mask 预处理，过滤离群小区域后再提取水线
        """
        mask = np.asarray(seg_mask)
        if mask.ndim != 2: return -1
        h, w = mask.shape
        
        # 提取水体二值掩码
        water_mask = (mask == self.water_class_id).astype(np.uint8)
        
        # 检查是否有水体
        if water_mask.sum() == 0:
            return -1
        
        # 【V7.1】可选的 mask 预处理
        if self.mask_preprocess:
            water_mask = self._clean_water_mask(water_mask)
            # 如果清理后没有水体，使用原始掩码
            if water_mask.sum() == 0:
                water_mask = (mask == self.water_class_id).astype(np.uint8)
        
        xmin, xmax = 0, w
        if x_range:
            xmin = max(0, int(x_range[0]))
            xmax = min(w, int(x_range[1]))
        if xmax <= xmin: xmin, xmax = 0, w
            
        tops = []
        # 步进采样加速
        for c in range(xmin, xmax, 2):
            rows = np.where(water_mask[:, c] > 0)[0]
            if len(rows) > 0: tops.append(rows.min())
            
        return float(np.median(tops)) if tops else -1

    def _regress_depth(self, scales, wl_proj, step):
        """
        深度回归 (V7.2优化: 局部加权回归 + 改进单点处理)
        
        仅使用离水线最近的若干真实刻度来拟合，降低透视畸变带来的非线性影响。
        
        拟合策略:
        - 单点 (len == 1): 使用经验步长公式 D = V_0 + k × (t_wl - ρ_0)
        - 多点 (len >= 2): 最小二乘法线性拟合 D = k × t_wl + b
        
        【V7.2改进】单点回归:
        - 多刻度场景: step 为靠近水线的分段加权平均步长
        - 单刻度场景: step 为经验值 avg_box_height × 1.8
        """
        if not scales: return None
        
        # 1) 优先使用真实检测点，数量不足则回退包含推断点
        real_scales = [s for s in scales if not s.is_inferred]
        if len(real_scales) < 2:
            real_scales = scales

        projs = np.array([s.proj_coord for s in real_scales])
        vals = np.array([s.value for s in real_scales])

        # 2) 局部拟合：取距离水线最近的 K 个点
        dists = np.abs(projs - wl_proj)
        K = 5
        if len(projs) > K:
            nearest_indices = np.argsort(dists)[:K]
            local_projs = projs[nearest_indices]
            local_vals = vals[nearest_indices]
        else:
            local_projs = projs
            local_vals = vals

        # 3) 拟合
        if len(local_projs) == 1:
            # 【V7.2】单点回归：使用最优搜索得到的局部步长
            # step 现在是 _build_grid_matrix 返回的加权平均步长（靠近水线的分段）
            if not step or step <= 0:
                step = 40.0  # 默认回退值
            
            # 理论斜率: k = -0.2m / S_opt (像素)
            # 物理含义: 每增加 S_opt 像素，深度减少 0.2m
            k_theory = -SCALE_INTERVAL / step
            
            # D = V_0 + k × (t_wl - ρ_0)
            d = local_vals[0] + k_theory * (wl_proj - local_projs[0])
            return max(0.0, float(d))

        # 多点情况：最小二乘拟合
        try:
            k, b = np.polyfit(local_projs, local_vals, 1)
            
            # 斜率校验：物理上深度应随投影坐标增大而减小 (k < 0)
            if k >= 0:
                # 局部拟合斜率异常，尝试全局拟合
                k_global, b_global = np.polyfit(projs, vals, 1)
                if k_global < 0:
                    return max(0.0, float(k_global * wl_proj + b_global))
                return None
            
            return max(0.0, float(k * wl_proj + b))
        except Exception:
            return None

    def _fail_result(self, wy, msg, dbg=None):
        return DepthEstimationResult(None, wy, None, [], "None", False, {"msg": msg, **(dbg or {})})


# =============================================================================
# 便捷函数 (Convenience Functions)
# =============================================================================

def quick_estimate_depth(
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    seg_mask: np.ndarray,
    class_names: List[str] = None,
    min_confidence: float = 0.3
) -> Optional[float]:
    """
    快速估计深度的便捷函数
    
    Args:
        boxes: 检测框 [N, 4], 格式 [x1, y1, x2, y2]
        labels: 类别标签 [N]
        scores: 置信度 [N]
        seg_mask: 分割掩码 H x W
        class_names: 类别名称列表
        min_confidence: 最小置信度
        
    Returns:
        深度值 (米) 或 None
    """
    estimator = StructureConstrainedVesselDraftDepthEstimation(class_names=class_names, min_confidence=min_confidence)
    result = estimator.estimate_with_details(boxes, labels, scores, seg_mask)
    return result.depth


def batch_estimate_depth(
    batch_data: List[Dict],
    class_names: List[str] = None,
    min_confidence: float = 0.3
) -> List[DepthEstimationResult]:
    """
    批量估计深度
    
    Args:
        batch_data: 每个元素包含 {'boxes', 'labels', 'scores', 'seg_mask'}
        class_names: 类别名称列表
        min_confidence: 最小置信度
        
    Returns:
        DepthEstimationResult 列表
    """
    estimator = StructureConstrainedVesselDraftDepthEstimation(class_names=class_names, min_confidence=min_confidence)
    results = []
    for data in batch_data:
        result = estimator.estimate_with_details(
            data['boxes'], data['labels'], data['scores'], data['seg_mask']
        )
        results.append(result)
    return results


def extract_waterline(
    seg_mask: np.ndarray,
    water_class_id: int = 1,
    x_range: Tuple[int, int] = None
) -> float:
    """
    从分割掩码提取水线位置
    
    Args:
        seg_mask: 分割掩码 H x W
        water_class_id: 水域类别ID
        x_range: X轴范围限制 (xmin, xmax)
        
    Returns:
        水线Y坐标，失败返回 -1
    """
    mask = np.asarray(seg_mask)
    if mask.ndim != 2:
        return -1
    h, w = mask.shape
    
    xmin, xmax = 0, w
    if x_range:
        xmin = max(0, int(x_range[0]))
        xmax = min(w, int(x_range[1]))
    if xmax <= xmin:
        xmin, xmax = 0, w
        
    tops = []
    for c in range(xmin, xmax, 2):
        rows = np.where(mask[:, c] == water_class_id)[0]
        if len(rows) > 0:
            tops.append(rows.min())
            
    return float(np.median(tops)) if tops else -1


# =============================================================================
# 单元测试
# =============================================================================
if __name__ == "__main__":
    print("Running StructureConstrainedVesselDraftDepthEstimation Self-Check (v4)...")
    
    # 模拟数据：垂直场景
    boxes_v = [[100, 100, 120, 120], [100, 150, 120, 170]] # 间隔50
    labels_v = [4, 6] # '5', '8' -> 5.8 (Err, just test logic)
    scores_v = [0.9, 0.9]
    mask_v = np.zeros((300, 300)); mask_v[200:, :] = 1
    
    est = StructureConstrainedVesselDraftDepthEstimation()
    res = est.estimate_with_details(boxes_v, labels_v, scores_v, mask_v)
    print(f"Vertical Test: Success={res.success}, Depth={res.depth}, WL_XY={res.waterline_xy}")
    
    # 模拟数据：倾斜场景 (dx=10, dy=50)
    boxes_t = [[100, 100, 120, 120], [110, 150, 130, 170]] 
    res_t = est.estimate_with_details(boxes_t, labels_v, scores_v, mask_v)
    print(f"Tilt Test: Success={res_t.success}, Depth={res_t.depth}, WL_XY={res_t.waterline_xy}")