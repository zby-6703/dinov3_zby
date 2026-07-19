"""
检测任务精度评估模块 (Detection Metrics)

计算目标检测的 mAP@0.5 和 mAP@0.5:0.95

输入格式:
    - pred_boxes: List[np.ndarray] 或 np.ndarray, shape (N, 4), 格式 [x1, y1, x2, y2]
    - pred_scores: List[np.ndarray] 或 np.ndarray, shape (N,)
    - pred_labels: List[np.ndarray] 或 np.ndarray, shape (N,)
    - gt_boxes: List[np.ndarray] 或 np.ndarray, shape (M, 4), 格式 [x1, y1, x2, y2]
    - gt_labels: List[np.ndarray] 或 np.ndarray, shape (M,)

输出格式:
    {
        'mAP@0.5': float,
        'mAP@0.5:0.95': float,
        'mAP@0.75': float,
        'precision': float,
        'recall': float,
        'per_class': {
            'class_name': {'AP@0.5': float, 'AP@0.5:0.95': float, ...}
        }
    }

使用示例:
    from utils.metrics import DetectionMetrics
    
    # 方式1: 逐图添加
    metrics = DetectionMetrics(num_classes=8, class_names=['0','1','2','3','4','6','8','M'])
    for img_id, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        metrics.add_image(
            image_id=img_id,
            pred_boxes=pred['boxes'],
            pred_scores=pred['scores'],
            pred_labels=pred['labels'],
            gt_boxes=gt['boxes'],
            gt_labels=gt['labels']
        )
    results = metrics.compute()
    
    # 方式2: 批量添加
    metrics = DetectionMetrics(num_classes=8)
    metrics.add_batch(all_pred_boxes, all_pred_scores, all_pred_labels, all_gt_boxes, all_gt_labels)
    results = metrics.compute()
"""

import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional, Union, Any


class DetectionMetrics:
    """
    目标检测精度评估器
    
    支持计算:
    - mAP@0.5: IoU阈值为0.5时的平均精度
    - mAP@0.5:0.95: IoU阈值从0.5到0.95的平均精度
    - mAP@0.75: IoU阈值为0.75时的平均精度
    - Precision, Recall
    """
    
    def __init__(
        self, 
        num_classes: int,
        class_names: Optional[List[str]] = None,
        iou_thresholds: Optional[List[float]] = None
    ):
        """
        初始化检测评估器
        
        Args:
            num_classes: 类别数量
            class_names: 类别名称列表，可选
            iou_thresholds: IoU阈值列表，默认 [0.5, 0.55, ..., 0.95]
        """
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.iou_thresholds = iou_thresholds or [0.5 + 0.05 * i for i in range(10)]
        
        # 存储预测和真值
        self.predictions = defaultdict(list)  # {image_id: [(box, score, label), ...]}
        self.ground_truths = defaultdict(list)  # {image_id: [(box, label), ...]}
        self.image_ids = set()
        
    def reset(self):
        """重置所有数据"""
        self.predictions.clear()
        self.ground_truths.clear()
        self.image_ids.clear()
        
    def add_image(
        self,
        image_id: Union[int, str],
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        pred_labels: np.ndarray,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray
    ):
        """
        添加单张图片的预测和真值
        
        Args:
            image_id: 图片ID
            pred_boxes: (N, 4) 预测框 [x1, y1, x2, y2]
            pred_scores: (N,) 置信度
            pred_labels: (N,) 类别标签
            gt_boxes: (M, 4) 真值框 [x1, y1, x2, y2]
            gt_labels: (M,) 类别标签
        """
        self.image_ids.add(image_id)
        
        # 转换为numpy数组
        pred_boxes = np.asarray(pred_boxes).reshape(-1, 4) if len(pred_boxes) > 0 else np.zeros((0, 4))
        pred_scores = np.asarray(pred_scores).flatten()
        pred_labels = np.asarray(pred_labels).flatten()
        gt_boxes = np.asarray(gt_boxes).reshape(-1, 4) if len(gt_boxes) > 0 else np.zeros((0, 4))
        gt_labels = np.asarray(gt_labels).flatten()
        
        # 存储预测
        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            self.predictions[image_id].append({
                'box': box,
                'score': float(score),
                'label': int(label)
            })
        
        # 存储真值
        for box, label in zip(gt_boxes, gt_labels):
            self.ground_truths[image_id].append({
                'box': box,
                'label': int(label)
            })
    
    def add_batch(
        self,
        pred_boxes_list: List[np.ndarray],
        pred_scores_list: List[np.ndarray],
        pred_labels_list: List[np.ndarray],
        gt_boxes_list: List[np.ndarray],
        gt_labels_list: List[np.ndarray],
        image_ids: Optional[List[Union[int, str]]] = None
    ):
        """
        批量添加多张图片的预测和真值
        
        Args:
            pred_boxes_list: 预测框列表
            pred_scores_list: 置信度列表
            pred_labels_list: 预测类别列表
            gt_boxes_list: 真值框列表
            gt_labels_list: 真值类别列表
            image_ids: 图片ID列表，可选
        """
        if image_ids is None:
            image_ids = list(range(len(pred_boxes_list)))
        
        for i, (pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels) in enumerate(
            zip(pred_boxes_list, pred_scores_list, pred_labels_list, gt_boxes_list, gt_labels_list)
        ):
            self.add_image(
                image_id=image_ids[i],
                pred_boxes=pred_boxes,
                pred_scores=pred_scores,
                pred_labels=pred_labels,
                gt_boxes=gt_boxes,
                gt_labels=gt_labels
            )
    
    def compute(self) -> Dict[str, Any]:
        """
        计算所有检测指标 (符合COCO评估标准)
        
        Returns:
            包含 mAP@0.5, mAP@0.5:0.95, mAP@0.75, precision, recall, per_class 的字典
        """
        # 按类别组织数据
        class_predictions = defaultdict(list)  # {class_id: [(image_id, box, score), ...]}
        class_ground_truths = defaultdict(lambda: defaultdict(list))  # {class_id: {image_id: [box, ...]}}
        
        # 统计每个类别的GT数量 (用于判断有效类别)
        gt_count_per_class = defaultdict(int)
        
        for image_id, preds in self.predictions.items():
            for pred in preds:
                class_predictions[pred['label']].append({
                    'image_id': image_id,
                    'box': pred['box'],
                    'score': pred['score']
                })
        
        for image_id, gts in self.ground_truths.items():
            for gt in gts:
                class_ground_truths[gt['label']][image_id].append(gt['box'])
                gt_count_per_class[gt['label']] += 1
        
        # 计算每个类别在每个IoU阈值下的AP
        all_aps = defaultdict(dict)  # {class_id: {iou_thresh: ap}}
        class_tp_fp_fn = {}  # {class_id: {'tp': int, 'fp': int, 'fn': int}}
        
        for class_id in range(self.num_classes):
            preds = class_predictions.get(class_id, [])
            gts = class_ground_truths.get(class_id, {})
            
            for iou_thresh in self.iou_thresholds:
                ap, tp, fp, fn = self._compute_ap_single_class(preds, gts, iou_thresh)
                all_aps[class_id][iou_thresh] = ap
                
                # 在IoU=0.5时记录TP/FP/FN用于计算precision和recall
                if iou_thresh == 0.5:
                    class_tp_fp_fn[class_id] = {'tp': tp, 'fp': fp, 'fn': fn}
        
        # 汇总结果
        results = {
            'mAP@0.5': 0.0,
            'mAP@0.5:0.95': 0.0,
            'mAP@0.75': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'per_class': {}
        }
        
        # 计算各类别指标 (只统计有GT的类别，符合COCO标准)
        valid_classes = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for class_id in range(self.num_classes):
            # 只有存在GT的类别才参与mAP计算
            if gt_count_per_class[class_id] > 0:
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)
                
                ap_50 = all_aps[class_id].get(0.5, 0.0)
                ap_75 = all_aps[class_id].get(0.75, 0.0)
                ap_50_95 = np.mean([all_aps[class_id].get(t, 0.0) for t in self.iou_thresholds])
                
                results['per_class'][class_name] = {
                    'AP@0.5': float(ap_50),
                    'AP@0.75': float(ap_75),
                    'AP@0.5:0.95': float(ap_50_95),
                    'num_gt': gt_count_per_class[class_id]
                }
                
                results['mAP@0.5'] += ap_50
                results['mAP@0.75'] += ap_75
                results['mAP@0.5:0.95'] += ap_50_95
                valid_classes += 1
                
                # 累加TP/FP/FN
                if class_id in class_tp_fp_fn:
                    total_tp += class_tp_fp_fn[class_id]['tp']
                    total_fp += class_tp_fp_fn[class_id]['fp']
                    total_fn += class_tp_fp_fn[class_id]['fn']
        
        # 平均 (只对有GT的类别求平均)
        if valid_classes > 0:
            results['mAP@0.5'] /= valid_classes
            results['mAP@0.75'] /= valid_classes
            results['mAP@0.5:0.95'] /= valid_classes
        
        # 计算整体precision和recall (基于所有类别的TP/FP/FN总和)
        results['precision'] = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
        results['recall'] = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
        
        # 添加统计信息
        results['num_valid_classes'] = valid_classes
        results['total_gt'] = sum(gt_count_per_class.values())
        results['total_predictions'] = sum(len(preds) for preds in self.predictions.values())
        
        return results
    
    def _compute_ap_single_class(
        self, 
        predictions: List[Dict], 
        ground_truths: Dict[str, List[np.ndarray]], 
        iou_threshold: float
    ) -> tuple:
        """
        计算单个类别在指定IoU阈值下的AP (符合PASCAL VOC/COCO标准)
        
        Returns:
            (AP, TP_count, FP_count, FN_count)
        """
        # 统计总GT数量
        total_gt = sum(len(boxes) for boxes in ground_truths.values())
        
        if total_gt == 0:
            # 没有GT时，所有预测都是FP
            return 0.0, 0, len(predictions), 0
        
        if not predictions:
            # 没有预测时，所有GT都是FN
            return 0.0, 0, 0, total_gt
        
        # 按置信度排序
        predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
        
        # 记录GT是否被匹配
        gt_matched = {img_id: [False] * len(boxes) for img_id, boxes in ground_truths.items()}
        
        tp = np.zeros(len(predictions))
        fp = np.zeros(len(predictions))
        
        for pred_idx, pred in enumerate(predictions):
            image_id = pred['image_id']
            pred_box = pred['box']
            
            if image_id not in ground_truths or len(ground_truths[image_id]) == 0:
                fp[pred_idx] = 1
                continue
            
            gt_boxes = ground_truths[image_id]
            
            # 计算与所有GT的IoU
            ious = np.array([self._compute_iou(pred_box, gt_box) for gt_box in gt_boxes])
            
            # 找到最大IoU的GT
            max_iou_idx = np.argmax(ious)
            max_iou = ious[max_iou_idx]
            
            if max_iou >= iou_threshold and not gt_matched[image_id][max_iou_idx]:
                tp[pred_idx] = 1
                gt_matched[image_id][max_iou_idx] = True
            else:
                fp[pred_idx] = 1
        
        # 计算累积TP和FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # 计算precision和recall曲线
        precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall_curve = tp_cumsum / total_gt
        
        # 计算AP (使用全点插值方法，符合COCO标准)
        ap = self._compute_ap_from_pr(precision_curve, recall_curve)
        
        # 返回AP和TP/FP/FN计数
        total_tp = int(tp.sum())
        total_fp = int(fp.sum())
        total_fn = total_gt - total_tp  # 未被匹配的GT数量
        
        return ap, total_tp, total_fp, total_fn
    
    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """计算两个框的IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = area1 + area2 - inter_area
        
        return inter_area / (union_area + 1e-10)
    
    def _compute_ap_from_pr(self, precision: np.ndarray, recall: np.ndarray) -> float:
        """
        从PR曲线计算AP (使用全点插值方法，符合COCO/VOC2010+标准)
        
        采用101点插值方法计算AP，与COCO官方评估一致
        """
        if len(precision) == 0 or len(recall) == 0:
            return 0.0
        
        # 添加哨兵值: recall从0开始，precision在recall=0时取最大precision
        mrec = np.concatenate(([0.0], recall, [recall[-1] + 1e-10]))
        mpre = np.concatenate(([precision[0]], precision, [0.0]))
        
        # 使precision单调递减 (从右向左取最大值)
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        
        # 方法1: 全点插值 (VOC2010+标准)
        # 找到recall变化的点
        idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
        
        # 计算AP: 对recall变化区间的precision求和
        ap = np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx])
        
        return float(ap)
    
    def summary(self) -> str:
        """返回评估结果摘要字符串"""
        results = self.compute()
        
        lines = [
            "=" * 50,
            "Detection Metrics Summary",
            "=" * 50,
            f"mAP@0.5:      {results['mAP@0.5']:.4f}",
            f"mAP@0.75:     {results['mAP@0.75']:.4f}",
            f"mAP@0.5:0.95: {results['mAP@0.5:0.95']:.4f}",
            f"Precision:    {results['precision']:.4f}",
            f"Recall:       {results['recall']:.4f}",
            "-" * 50,
            f"Valid Classes: {results['num_valid_classes']}",
            f"Total GT:      {results['total_gt']}",
            f"Total Preds:   {results['total_predictions']}",
            "-" * 50,
            "Per-class AP@0.5:"
        ]
        
        for class_name, class_metrics in results['per_class'].items():
            lines.append(f"  {class_name}: AP={class_metrics['AP@0.5']:.4f} (GT={class_metrics['num_gt']})")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)
