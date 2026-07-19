"""
船舶吃水深度估计 - 论文复现基准模型 (Baseline)

复现论文: "Multi-Task Learning-Enabled Automatic Vessel Draft Reading for Intelligent Maritime Surveillance"
(IEEE TITS, 2024)

核心算法实现:
    1. Algorithm 1: Draft Scale Recognition and Correction
       - Step 1: Cross-class NMS
       - Step 2: Draft scale recognition (Character Association)
       - Step 3: Recognized results scoring (Spatial Rule Validation)
       - Step 4: Mistaken recognition correction (Linear Interpolation)
    2. Draft Depth Estimation
       - Equation 10: Dynamic adaptive linear fitting (Multi-scale)
       - Equation 11: Character height-based estimation (Single-scale)

输入:
    - detection_boxes: [x1, y1, x2, y2]
    - detection_labels: 类别 (0-9, M)
    - detection_scores: 置信度
    - segmentation_mask: 水体掩码

输出:
    - estimated_depth: 吃水深度
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

# =============================================================================
# 常量定义 (基于论文)
# =============================================================================

SCALE_INTERVAL = 0.2        # 相邻刻度数值差 (米)
REAL_CHAR_HEIGHT = 0.1      # 字符物理高度 (米) - Eq.11 中的 beta
MAX_DIST_RATIO = 2.3        # 评分规则：相邻刻度像素距离 < 2.3 * 字符高度
IOU_THRESHOLD = 0.3         # NMS IoU 阈值

# 水线提取预处理参数
MASK_PREPROCESS_ENABLED = True          # 是否启用 mask 预处理（可选，用于对比实验）
MASK_MORPH_KERNEL_SIZE = 5              # 形态学操作核大小
MASK_MIN_AREA_RATIO = 0.01              # 最小连通区域面积比例（相对于图像面积）

# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class BaselineScale:
    """论文算法中的刻度对象"""
    xc: float       # 中心x
    yc: float       # 中心y
    w: float        # 宽
    h: float        # 高
    value: float    # 解析后的数值 (如 3.2)
    score: int      # 0 or 1
    raw_conf: float # 原始置信度(取均值)

@dataclass
class ScaleReading:
    """为了兼容现有评测接口的输出结构"""
    value: float
    y_center: float
    score: int
    is_inferred: bool = False

@dataclass
class DepthEstimationResult:
    """标准输出结果"""
    depth: Optional[float]
    waterline_y: float
    scales: List[ScaleReading]
    method: str
    success: bool
    debug_info: Dict = field(default_factory=dict)

# =============================================================================
# 核心类：论文算法复现
# =============================================================================

class BaselineDepthEstimator:
    
    # 论文中涉及的类别
    DEFAULT_CLASS_NAMES = ['0', '1', '2', '3', '4', '6', '8', 'M', '5', '7', '9']

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        water_class_id: int = 1,
        min_confidence: float = 0.3, # 论文中未明确提及阈值，设默认值
        mask_preprocess: Optional[bool] = None
    ):
        """
        初始化基准深度估计器
        
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
        self.mask_preprocess = mask_preprocess if mask_preprocess is not None else MASK_PREPROCESS_ENABLED

    def estimate(self, det_boxes, det_labels, det_scores, seg_mask) -> Optional[float]:
        res = self.estimate_with_details(det_boxes, det_labels, det_scores, seg_mask)
        return res.depth

    def estimate_with_details(
        self,
        det_boxes: np.ndarray,
        det_labels: np.ndarray,
        det_scores: np.ndarray,
        seg_mask: np.ndarray
    ) -> DepthEstimationResult:
        
        # --- 0. 数据预处理 ---
        # 将输入转换为论文算法需要的格式: [Bbox, Class, Conf]
        raw_inputs = self._preprocess_inputs(det_boxes, det_labels, det_scores)
        
        # --- 1. Algorithm 1 Step 1: Cross-class NMS ---
        # 论文 Step 1: 基于空间分布规则的 NMS
        nms_boxes = self._step1_cross_class_nms(raw_inputs)
        
        # --- 2. Algorithm 1 Step 2: Draft Scale Recognition ---
        # 论文 Step 2: 字符关联 (Association) 生成刻度
        recognized_scales = self._step2_recognition(nms_boxes)
        
        # --- 3. Algorithm 1 Step 3: Scoring ---
        # 论文 Step 3: 基于 0.2m 间隔和 2.3h 距离的评分
        scored_scales = self._step3_scoring(recognized_scales)
        
        # --- 4. Algorithm 1 Step 4: Correction ---
        # 论文 Step 4: 对 score=0 的刻度进行线性插值修正
        final_scales = self._step4_correction(scored_scales)
        
        # --- 5. Draft Depth Estimation ---
        # 提取水线
        waterline_y = self._extract_waterline(seg_mask)
        
        # 计算深度 (Eq 10 & 11)
        depth, method = self._calculate_depth(final_scales, waterline_y)
        
        # --- 6. 格式化输出 ---
        output_scales = [
            ScaleReading(s.value, s.yc, s.score, is_inferred=(s.score==2)) 
            for s in final_scales
        ]
        
        return DepthEstimationResult(
            depth=depth,
            waterline_y=waterline_y,
            scales=output_scales,
            method=f"MTL-VDR-Baseline ({method})",
            success=(depth is not None),
            debug_info={"raw_count": len(raw_inputs), "nms_count": len(nms_boxes)}
        )

    # =========================================================================
    # 算法实现细节
    # =========================================================================

    def _preprocess_inputs(self, boxes, labels, scores):
        """转换输入格式"""
        data = []
        if len(boxes) == 0: return data
        
        boxes = np.array(boxes).reshape(-1, 4)
        labels = np.array(labels).flatten()
        scores = np.array(scores).flatten()
        
        for b, l, s in zip(boxes, labels, scores):
            if s < self.min_confidence: continue
            l = int(l)
            if l >= len(self.class_names): continue
            
            cls_name = self.class_names[l]
            if cls_name == 'water': continue
            
            # Bbox format: [x, y, w, h] (center x, center y, width, height)
            x1, y1, x2, y2 = b
            w, h = x2 - x1, y2 - y1
            xc, yc = (x1 + x2)/2, (y1 + y2)/2
            
            data.append({
                'bbox': [xc, yc, w, h], # Paper uses center format
                'class': cls_name,
                'conf': float(s)
            })
        return data

    def _iou(self, box1, box2):
        """计算两个中心点格式box的IoU"""
        # box: [xc, yc, w, h]
        b1_x1, b1_x2 = box1[0] - box1[2]/2, box1[0] + box1[2]/2
        b1_y1, b1_y2 = box1[1] - box1[3]/2, box1[1] + box1[3]/2
        b2_x1, b2_x2 = box2[0] - box2[2]/2, box2[0] + box2[2]/2
        b2_y1, b2_y2 = box2[1] - box2[3]/2, box2[1] + box2[3]/2

        inter_x1 = max(b1_x1, b2_x1)
        inter_y1 = max(b1_y1, b2_y1)
        inter_x2 = min(b1_x2, b2_x2)
        inter_y2 = min(b1_y2, b2_y2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        b1_area = box1[2] * box1[3]
        b2_area = box2[2] * box2[3]
        
        union = b1_area + b2_area - inter_area + 1e-6
        return inter_area / union

    def _step1_cross_class_nms(self, boxes: List[Dict]) -> List[Dict]:
        """
        Algorithm 1 Step 1: Cross-class non-maximum suppression
        Lines 8-11
        """
        if not boxes: return []
        
        # 标记需要删除的索引
        to_delete = set()
        n = len(boxes)
        
        for i in range(n):
            if i in to_delete: continue
            for j in range(n):
                if i == j or j in to_delete: continue
                
                # IoU > 0.3 and Conf_i > Conf_j -> Delete j
                iou_val = self._iou(boxes[i]['bbox'], boxes[j]['bbox'])
                if iou_val > IOU_THRESHOLD:
                    if boxes[i]['conf'] > boxes[j]['conf']:
                        to_delete.add(j)
                    else:
                        to_delete.add(i)
                        break # i is deleted, stop checking i
        
        return [b for k, b in enumerate(boxes) if k not in to_delete]

    def _step2_recognition(self, boxes: List[Dict]) -> List[BaselineScale]:
        """
        Algorithm 1 Step 2: Draft scale recognition
        Lines 12-19
        Merging characters into scales based on proximity
        """
        # 为了避免重复合并，先按y排序
        boxes.sort(key=lambda x: x['bbox'][1])
        used = [False] * len(boxes)
        scales = []
        
        for i in range(len(boxes)):
            if used[i]: continue
            
            merged = False
            bi = boxes[i]
            xi, yi, wi, hi = bi['bbox']
            
            # 寻找可以合并的邻居 (lines 13-14)
            for j in range(len(boxes)):
                if i == j or used[j]: continue
                
                bj = boxes[j]
                xj, yj, wj, hj = bj['bbox']
                
                # Condition: 0 < xj - xi < 2wi (水平相邻)
                # and |yj - yi| < min(hi, hj) (垂直对齐)
                # 注意：这里假设 xj > xi，即 j 在 i 右边。如果 j 在 i 左边，逻辑需反转
                # 论文伪代码写的是 xj - xi，暗示了顺序。我们做个简单的左右判断。
                
                # 确保 box_left 和 box_right
                if xi < xj:
                    left, right = bi, bj
                    idx_l, idx_r = i, j
                else:
                    left, right = bj, bi
                    idx_l, idx_r = j, i
                
                lx, ly, lw, lh = left['bbox']
                rx, ry, rw, rh = right['bbox']
                
                # 论文条件复现
                h_cond = (0 < (rx - lx) < 2 * lw)
                v_cond = (abs(ry - ly) < min(lh, rh))
                
                if h_cond and v_cond:
                    # Merge (Lines 15-18)
                    xc = (lx + rx) / 2
                    yc = (ly + ry) / 2
                    hc = (lh + rh) / 2
                    wc = (lw + rw) # 宽度累加近似
                    
                    # String concatenation C(Class_i; Class_j)
                    # Replacing 'M' with '0' -> handled in _parse_value
                    val = self._parse_value(left['class'], right['class'])
                    
                    scales.append(BaselineScale(
                        xc=xc, yc=yc, w=wc, h=hc,
                        value=val, score=1, # 初始设为1，Step 3会重置
                        raw_conf=(left['conf'] + right['conf'])/2
                    ))
                    used[idx_l] = True
                    used[idx_r] = True
                    merged = True
                    break
            
            if not merged:
                # 无法合并的单个字符
                # 尝试将其本身解析为刻度 (例如单个数字无法解析为x.x，除非是M)
                # 论文逻辑似乎主要针对"数字+数字"或"数字+M"。
                # 为了鲁棒性，如果是单字符 'M'，可视作 X.0 (但缺乏整数位信息)。
                # 或者单数字，可能对应 X.X。
                # 此处保守处理：尝试解析单字符，如果是 'M' 暂存，如果是数字暂存
                # 但根据论文公式 (17行)，它是除以10。
                # 只有两个字符才能除以10得到小数。
                # 暂时跳过单字符，或者视情况处理。
                # 实际上单字符很难确定其数值，除非有上下文。
                pass

        # 按 Y 坐标排序 (Top to Bottom) -> Value Large to Small
        scales.sort(key=lambda s: s.yc)
        return scales

    def _step3_scoring(self, scales: List[BaselineScale]) -> List[BaselineScale]:
        """
        Algorithm 1 Step 3: Recognized results scoring
        Lines 20-24
        """
        n = len(scales)
        if n == 0: return []
        
        # 默认全部初始化为 0 (Line 24 logic inverted for initialization)
        for s in scales: s.score = 0
        
        for i in range(n):
            ci = scales[i].value
            yi = scales[i].yc
            hi = scales[i].h
            
            # Check previous (above)
            valid_prev = False
            if i > 0:
                c_prev = scales[i-1].value
                y_prev = scales[i-1].yc
                # Cond 1: abs(ci - c_prev) == 0.2
                # Cond 2: yi - y_prev < 2.3 * hi
                if abs(c_prev - ci - 0.2) < 0.05 and (yi - y_prev) < MAX_DIST_RATIO * hi:
                    valid_prev = True
            
            # Check next (below)
            valid_next = False
            if i < n - 1:
                c_next = scales[i+1].value
                y_next = scales[i+1].yc
                # Cond 1: abs(ci - c_next) == 0.2 (Assuming sorted large to small value, small y to large y)
                # Wait, scales sorted by Y (small y = top = large value).
                # So c_next should be smaller. ci - c_next = 0.2
                if abs(ci - c_next - 0.2) < 0.05 and (y_next - yi) < MAX_DIST_RATIO * hi:
                    valid_next = True
            
            if valid_prev or valid_next:
                scales[i].score = 1
                
        return scales

    def _step4_correction(self, scales: List[BaselineScale]) -> List[BaselineScale]:
        """
        Algorithm 1 Step 4: Mistaken recognition correction
        Lines 25-33
        """
        # 找出 score=1 的作为参考
        correct_scales = [s for s in scales if s.score == 1]
        if len(correct_scales) < 2:
            # 少于2个正确刻度，无法进行 d1 计算 (yN - yL)，无法插值
            # 只能返回原始有的正确刻度
            return correct_scales
        
        # 按 Y 排序 (Value Large to Small)
        correct_scales.sort(key=lambda s: s.yc)
        
        for i in range(len(scales)):
            if scales[i].score == 0:
                s_i = scales[i]
                yi = s_i.yc
                
                # Search nearest two correct scales (DL, DN)
                # F(Dcorrect, Di) -> [DL, DN]
                # DL: Upper (Larger value, Smaller Y)
                # DN: Lower (Smaller value, Larger Y)
                
                # 找到 yi 上方最近的 (y < yi)
                DL = None
                for cs in reversed(correct_scales):
                    if cs.yc < yi:
                        DL = cs
                        break
                
                # 找到 yi 下方最近的 (y > yi)
                DN = None
                for cs in correct_scales:
                    if cs.yc > yi:
                        DN = cs
                        break
                
                if DL and DN:
                    yL, cL = DL.yc, DL.value
                    yN, cN = DN.yc, DN.value
                    
                    d1 = yN - yL
                    d2 = yi - yL
                    
                    # Eq 9: ci = phi(cL - d2/d1 * (cL - cN))
                    pred_val = cL - (d2 / d1) * (cL - cN)
                    
                    # phi(x): find closest integral multiple of 0.2
                    corrected_val = round(pred_val / 0.2) * 0.2
                    
                    # 避免重复: check if corrected_val exists in correct_scales
                    exists = any(abs(cs.value - corrected_val) < 0.05 for cs in correct_scales)
                    
                    if not exists:
                        s_i.value = corrected_val
                        s_i.score = 2 # 标记为 Corrected
                        # 暂时不加入 correct_scales 列表以避免迭代影响
        
        return scales

    def _clean_water_mask(self, water_mask: np.ndarray) -> np.ndarray:
        """
        清理水体掩码，过滤离群小区域
        
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

    def _extract_waterline(self, seg_mask) -> float:
        """
        从分割掩码提取水线Y坐标
        
        可选的 mask 预处理，过滤离群小区域后再提取水线
        """
        if seg_mask is None: return -1
        
        # 提取水体二值掩码
        water_mask = (seg_mask == self.water_class_id).astype(np.uint8)
        if water_mask.sum() == 0: return -1
        
        # 可选的 mask 预处理
        if self.mask_preprocess:
            water_mask = self._clean_water_mask(water_mask)
            if water_mask.sum() == 0:
                water_mask = (seg_mask == self.water_class_id).astype(np.uint8)
        
        # 论文中使用 "top pixels of water"
        # 这里取每一列的最上方像素，然后取中位数
        H, W = seg_mask.shape
        top_ys = []
        for c in range(0, W, 5): # 采样
            rows = np.where(water_mask[:, c] > 0)[0]
            if len(rows) > 0:
                top_ys.append(rows.min())
        
        if not top_ys: return -1
        return float(np.median(top_ys))

    def _calculate_depth(self, scales: List[BaselineScale], waterline_y: float) -> Tuple[Optional[float], str]:
        """
        Draft Depth Estimation
        Eq 10 (Multi-scale) & Eq 11 (Single-scale)
        """
        if not scales or waterline_y < 0:
            return None, "No Scales/Waterline"
        
        # 过滤掉水线以下的刻度 (倒影) - 论文虽然没明说，但物理上必须如此
        # Y 轴向下增大。水线 Yw。刻度 Ys。
        # 有效刻度应在水线上方 => Ys < Yw
        valid_scales = [s for s in scales if s.yc < waterline_y]
        
        if not valid_scales:
            return None, "No Scales Above Waterline"
        
        # Sort by Y (Closest to waterline is the last one in list, largest Y)
        valid_scales.sort(key=lambda s: s.yc)
        S_closest = valid_scales[-1] # S1 or S3
        
        # d: relative distance of water to closest scale (positive)
        d = waterline_y - S_closest.yc
        
        # Scenario A: Multiple scales available (Eq 10)
        # S1 (Closest), S2 (Neighbor above S1)
        if len(valid_scales) >= 2:
            S_neighbor = valid_scales[-2] # S2
            
            d1 = S_closest.yc - S_neighbor.yc # Distance between scales
            val_diff = S_neighbor.value - S_closest.value # Should be 0.2 usually
            
            # Eq 10: D = d/d1 * (S1 - S2) ? 
            # Wait, Eq 10 in paper: D = d/d1 * (S1 - S2)
            # This calculates the *increment* depth below S1.
            # Total Depth = S1_value - increment?
            # 论文公式 (10) D = d/d1(S1-S2) 计算的是吃水深度值吗？
            # 实际上，吃水深度在水线处。
            # S1=3.2m, S2=3.4m. S1在下，S2在上。
            # 吃水深度应该是 3.2 - delta.
            # 论文中 S1 是 closest scale. S2 是 neighbor.
            # 如果 S1=3.2, S2=3.4. (S1-S2) = -0.2.
            # D = ratio * (-0.2). 负值。
            # 这似乎是相对于 S1 的偏移量？
            # 结合 Fig 6，可以推断最终深度是 S1 - ratio * abs(S1-S2)。
            # 或者如果是 S1=3.2, S2=3.0 (S2在S1下方? 不可能，S1最接近水线，S2只能在上方)
            # 让我们假设标准线性插值：
            # Depth = S_closest.value - (d / d1) * abs(S_neighbor.value - S_closest.value)
            
            # 修正：Draft Value 随 Y 增大而减小。
            # S_neighbor (top, small y) -> 3.4m
            # S_closest (bottom, large y) -> 3.2m
            # Water (larger y) -> Depth < 3.2m
            # d = WaterY - ClosestY (Pos)
            # d1 = ClosestY - NeighborY (Pos)
            # ratio = d / d1
            # Depth = 3.2 - ratio * (3.4 - 3.2) = 3.2 - ratio * 0.2
            
            depth = S_closest.value - (d / d1) * abs(S_neighbor.value - S_closest.value)
            return float(depth), "Eq.10 (Multi)"
            
        # Scenario B: Single scale available (Eq 11)
        else:
            # D = S3 - (beta * d) / h1
            # S3 = S_closest.value
            # beta = 0.1 (REAL_CHAR_HEIGHT)
            # h1 = S_closest.h (Pixel height)
            
            depth = S_closest.value - (REAL_CHAR_HEIGHT * d) / S_closest.h
            return float(depth), "Eq.11 (Single)"

    def _parse_value(self, c1, c2):
        """将两个字符转换为数值: '3', 'M' -> 3.0; '3', '2' -> 3.2"""
        # Replace 'M' with '0'
        d1 = '0' if c1 == 'M' else c1
        d2 = '0' if c2 == 'M' else c2
        
        try:
            # 论文是将字符串拼接然后除以10
            # C(Class_i; Class_j) / 10
            combined = float(d1 + d2)
            return combined / 10.0
        except:
            return 0.0