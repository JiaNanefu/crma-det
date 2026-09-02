"""
Enhanced Detection Engine with YOLO-style logging
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modified by lyuwenyu
"""

import math
import os
import sys
import time
from contextlib import redirect_stdout
from typing import Iterable
import numpy as np

import torch

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from src.data import CocoEvaluator
from src.misc import (MetricLogger, SmoothedValue, reduce_dict, dist)


def format_time(seconds):
    """格式化时间为 mm:ss 或 hh:mm:ss"""
    if seconds < 3600:
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _detect_enabled_losses(loss_dict):
    """从 criterion 返回的 loss_dict 中检测启用的 base loss 项（忽略 aux/dn 层）
    
    返回排序后的 loss 键列表，按照偏好顺序排列
    """
    def is_base_loss(k):
        if not isinstance(k, str) or not k.startswith('loss_'):
            return False
        if '_aux_' in k or '_dn_' in k:
            return False
        if 'accuracy' in k or 'cardinality' in k:
            return False
        return True
    
    base_keys = [k for k in loss_dict.keys() if is_base_loss(k)]
    
    # 按偏好顺序排列
    preferred_order = [
        'loss_vfl',
        'loss_ordinal',
        'loss_ord_branch',
        'loss_bbox',
        'loss_giou',
        'loss_dfl',
    ]
    ordered = [k for k in preferred_order if k in base_keys]
    ordered += [k for k in base_keys if k not in preferred_order]
    
    return ordered


def _infer_loss_keys_from_criterion(criterion):
    """从 criterion 配置中推断可能的 loss 键（不需要实际前向传播）
    
    这个方法用于在第一个 batch 之前预测 loss 列名，以便打印表头
    """
    loss_keys = []
    
    # 检查是否使用 ordinal 还是 vfl
    use_ordinal = getattr(criterion, 'use_ordinal', False)
    if use_ordinal:
        loss_keys.append('loss_ordinal')
    else:
        loss_keys.append('loss_vfl')

    if 'ord_branch' in getattr(criterion, 'losses', []):
        loss_keys.append('loss_ord_branch')
    
    # bbox 和 giou 通常始终存在
    loss_keys.extend(['loss_bbox', 'loss_giou'])
    
    # DFL 取决于 use_dfl 或 use_fdr
    use_dfl = getattr(criterion, 'use_dfl', False) or getattr(criterion, 'use_fdr', False)
    if use_dfl:
        loss_keys.append('loss_dfl')
    
    return loss_keys


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    """训练一个epoch，使用YOLO风格的tqdm进度条
    
    动态识别启用的 loss 项，仅显示/记录实际启用的 loss
    格式参考 (动态列):
    Epoch    GPU_mem   loss_vfl  loss_bbox  loss_giou  Instances       Size
    278/300      4.46G     0.4351     0.5273     0.9428         47        640: 100%|██████████| 100/100 [00:21<00:00,  4.62it/s]
    """
    model.train()
    criterion.train()
    
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    total_epochs = kwargs.get('total_epochs', 300)
    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)
    
    total_batches = len(data_loader)
    
    # 暂存推断的 loss 键，第一个 batch 后用实际键覆盖
    enabled_loss_keys = _infer_loss_keys_from_criterion(criterion)
    num_instances = 0
    
    # 创建tqdm进度条
    if tqdm is not None and dist.is_main_process():
        pbar = tqdm(
            enumerate(data_loader),
            total=total_batches,
            bar_format='{l_bar}{bar:10}{r_bar}',
            mininterval=1.0,  # 每秒最多刷新一次
            leave=True,
            ncols=None  # 自动适应终端宽度
        )
    else:
        pbar = enumerate(data_loader)
    
    # 用于检测实际 loss 键（第一个 batch 后更新）
    actual_loss_keys_updated = False
    loss_col_width = 12     # 动态列宽，第一 batch 后根据实际列名自动调整
    
    # 训练循环
    for batch_idx, (samples, targets) in pbar:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # 计算当前batch的实例数
        num_instances = sum(len(t['labels']) for t in targets)

        # 前向传播和反向传播
        if scaler is not None:
            with torch.amp.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets)
            
            with torch.amp.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
        
        # EMA更新
        if ema is not None:
            ema.update(model)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())
        
        # 第一个 batch 后，用实际 loss_dict 确定 loss 键并打印表头
        if not actual_loss_keys_updated:
            enabled_loss_keys = _detect_enabled_losses(loss_dict_reduced)

            # 动态计算列宽：取表头名和数值格式(最小12)中较长者
            loss_col_width = max(12, max((len(k.replace('loss_', '') + '_loss') for k in enabled_loss_keys), default=12) + 2)

            # 打印动态表头（此时已知所有实际 loss 键，与后续进度条严格对齐）
            if dist.is_main_process():
                # 清除 tqdm 第一次迭代时渲染的 0% 行，避免表头前残留进度条
                if hasattr(pbar, 'clear'):
                    pbar.clear()
                header_parts = [f"{'Epoch':>10}", f"{'GPU_mem':>12}"]
                for key in enabled_loss_keys:
                    display_name = key.replace('loss_', '') + '_loss'
                    header_parts.append(f"{display_name:>{loss_col_width}}")
                header_parts.extend([f"{'Instances':>12}", f"{'Size':>10}"])
                print("".join(header_parts))

            actual_loss_keys_updated = True

        # 使用 .item() 避免梯度图保留警告
        if not math.isfinite(loss_value.item() if torch.is_tensor(loss_value) else loss_value):
            print("Loss is {}, stopping training".format(loss_value.item() if torch.is_tensor(loss_value) else loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # 使用 .item() 转换为标量，避免保留梯度图
        loss_value_scalar = loss_value.item() if torch.is_tensor(loss_value) else loss_value
        loss_dict_reduced_scalar = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict_reduced.items()}
        
        metric_logger.update(loss=loss_value_scalar, **loss_dict_reduced_scalar)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        # 更新tqdm进度条描述 (动态列)
        if tqdm is not None and dist.is_main_process():
            gpu_mem = f"{torch.cuda.memory_reserved() / 1E9:.2f}G" if torch.cuda.is_available() else "N/A"
            epoch_str = f"{epoch+1}/{total_epochs}"
            
            desc_parts = [f"{epoch_str:>10}", f"{gpu_mem:>12}"]
            for key in enabled_loss_keys:
                val = loss_dict_reduced_scalar.get(key, 0)
                desc_parts.append(f"{val:>{loss_col_width}.4f}")
            desc_parts.extend([f"{num_instances:>12}", f"{640:>10}"])
            
            pbar.set_description("".join(desc_parts))
    
    # 关闭进度条
    if tqdm is not None and dist.is_main_process() and hasattr(pbar, 'close'):
        pbar.close()
    
    # 计算平均损失
    metric_logger.synchronize_between_processes()
    avg_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    return avg_stats


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, 
             data_loader, base_ds, device, output_dir, epoch=-1, total_epochs=-1, is_test=False):
    """评估模型，使用YOLO风格的tqdm进度条"""
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")

    iou_types = postprocessors.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    total_batches = len(data_loader)
    
    # YOLO风格：打印验证表头
    if dist.is_main_process():
        header = f"{'Phase':>10}{'Instances':>12}{'Size':>10}"
        print(header)
    
    # 创建tqdm进度条
    if tqdm is not None and dist.is_main_process():
        pbar = tqdm(
            enumerate(data_loader),
            total=total_batches,
            bar_format='{l_bar}{bar:10}{r_bar}',
            mininterval=1.0,  # 每秒最多刷新一次
            leave=True,
            ncols=None  # 自动适应终端宽度
        )
    else:
        pbar = enumerate(data_loader)
    
    for batch_idx, (samples, targets) in pbar:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        
        # 致命Bug修复：必须在PostProcessor之前计算Loss！
        # Loss函数需要归一化坐标[0,1]，而PostProcessor会转换为绝对坐标
        loss_dict = criterion(outputs, targets)
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())
        
        # 转换为标量
        loss_value_scalar = loss_value.item() if torch.is_tensor(loss_value) else loss_value
        loss_dict_reduced_scalar = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict_reduced.items()}
        
        metric_logger.update(loss=loss_value_scalar, **loss_dict_reduced_scalar)

        # PostProcessor：将归一化坐标转换为绝对坐标用于COCO评估
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        
        results = postprocessors(outputs, orig_target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)
        
        # 更新tqdm进度条描述
        if tqdm is not None and dist.is_main_process():
            num_instances = sum(len(t['labels']) for t in targets)
            desc = f"{'val':>10}{num_instances:>12}{640:>10}"
            pbar.set_description(desc)
    
    # 关闭进度条
    if tqdm is not None and dist.is_main_process() and hasattr(pbar, 'close'):
        pbar.close()

    # 同步和累积结果
    metric_logger.synchronize_between_processes()
    category_stats = {}  # 初始化，避免未定义错误
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        # 静音 COCO 默认输出，使用自定义表格展示
        with redirect_stdout(open(os.devnull, 'w')):
            coco_evaluator.accumulate()
            coco_evaluator.summarize()  # 关键修复：必须调用 summarize() 才能填充 stats 数组
        
        # YOLO风格的类别统计输出，并获取每个类别的指标
        category_stats = print_yolo_style_metrics(coco_evaluator, base_ds, is_test)

    # Bug 3修复：返回验证损失
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    # 合并每个类别的指标到stats中
    if category_stats:
        stats.update(category_stats)
    
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            bbox_stats = coco_evaluator.coco_eval['bbox'].stats
            # 处理 stats 可能是 numpy 数组或列表的情况
            if hasattr(bbox_stats, 'tolist'):
                stats['coco_eval_bbox'] = bbox_stats.tolist()
            else:
                stats['coco_eval_bbox'] = list(bbox_stats) if bbox_stats is not None else []

    return stats, coco_evaluator


def print_yolo_style_metrics(coco_evaluator, base_ds, is_test=False):
    """YOLOv8风格的指标输出，返回每个类别的指标字典"""
    if not dist.is_main_process():
        return {}
    
    coco_eval = coco_evaluator.coco_eval['bbox']
    category_stats = {}  # 存储每个类别的指标
    
    # 获取所有类别
    cat_ids = base_ds.getCatIds()
    cats = base_ds.loadCats(cat_ids)
    
    # 检查评估结果是否有效
    if coco_eval is None or coco_eval.eval is None:
        # 早期epoch可能没有检测到任何对象，显示0值
        all_images = len(coco_eval.params.imgIds) if coco_eval and coco_eval.params else 0
        all_instances = 0
        for cat in cats:
            cat_instances = 0
            if coco_eval and coco_eval.params:
                for img_id in coco_eval.params.imgIds:
                    ann_ids = base_ds.getAnnIds(imgIds=img_id, catIds=cat['id'])
                    cat_instances += len(ann_ids)
            all_instances += cat_instances
        
        # YOLOv8格式表头
        print(f"{'Class':>15} {'Images':>8} {'Instances':>10} {'Box(P':>10} {'R':>10} {'mAP50':>10} {'mAP50-95)':>12}")
        print(f"{'all':>15} {all_images:>8} {all_instances:>10} {0:>10.4f} {0:>10.4f} {0:>10.4f} {0:>12.4f}")
        for cat in cats:
            cat_name = cat['name']
            cat_instances = 0
            if coco_eval and coco_eval.params:
                for img_id in coco_eval.params.imgIds:
                    ann_ids = base_ds.getAnnIds(imgIds=img_id, catIds=cat['id'])
                    cat_instances += len(ann_ids)
            if cat_instances > 0:
                print(f"{cat_name:>15} {all_images:>8} {cat_instances:>10} {0:>10.4f} {0:>10.4f} {0:>10.4f} {0:>12.4f}")
                # 记录每个类别的指标（全0）
                category_stats[f'metrics/{cat_name}_P'] = 0.0
                category_stats[f'metrics/{cat_name}_R'] = 0.0
                category_stats[f'metrics/{cat_name}_mAP50'] = 0.0
                category_stats[f'metrics/{cat_name}_mAP50-95'] = 0.0
        return category_stats
    
    # YOLOv8格式表头
    print(f"{'Class':>15} {'Images':>8} {'Instances':>10} {'Box(P':>10} {'R':>10} {'mAP50':>10} {'mAP50-95)':>12}")
    
    # 计算每个类别的指标
    precision = coco_eval.eval.get('precision', None)  # [T, R, K, A, M]
    recall = coco_eval.eval.get('recall', None)  # [T, K, A, M]
    scores = coco_eval.eval.get('scores', None)  # [T, R, K, A, M]
    
    if precision is None or recall is None:
        # 显示0值
        all_images = len(coco_eval.params.imgIds)
        all_instances = 0
        for cat in cats:
            cat_instances = 0
            for img_id in coco_eval.params.imgIds:
                ann_ids = base_ds.getAnnIds(imgIds=img_id, catIds=cat['id'])
                cat_instances += len(ann_ids)
            all_instances += cat_instances
        
        print(f"{'all':>15} {all_images:>8} {all_instances:>10} {0:>10.4f} {0:>10.4f} {0:>10.4f} {0:>12.4f}")
        for cat in cats:
            cat_name = cat['name']
            cat_instances = 0
            for img_id in coco_eval.params.imgIds:
                ann_ids = base_ds.getAnnIds(imgIds=img_id, catIds=cat['id'])
                cat_instances += len(ann_ids)
            if cat_instances > 0:
                print(f"{cat_name:>15} {all_images:>8} {cat_instances:>10} {0:>10.4f} {0:>10.4f} {0:>10.4f} {0:>12.4f}")
                # 记录每个类别的指标（全0）
                category_stats[f'metrics/{cat_name}_P'] = 0.0
                category_stats[f'metrics/{cat_name}_R'] = 0.0
                category_stats[f'metrics/{cat_name}_mAP50'] = 0.0
                category_stats[f'metrics/{cat_name}_mAP50-95'] = 0.0
        return category_stats
    
    # precision: [iou_thresholds, recall_thresholds, categories, area_ranges, max_dets]
    # recall: [iou_thresholds, categories, area_ranges, max_dets]
    
    # 获取所有类别的统计
    all_images = len(coco_eval.params.imgIds)
    all_instances = 0
    
    # 建立类别ID到precision数组索引的映射
    # coco_eval.params.catIds 是 precision 数组中类别维度的顺序
    eval_cat_ids = coco_eval.params.catIds  # 评估器使用的类别ID列表
    cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(eval_cat_ids)}
    
    # Bug 4修复：防止 IndexError，增加判空逻辑
    stats = coco_eval.stats
    if stats is not None and hasattr(stats, '__len__') and len(stats) > 1:
        try:
            overall_map50_95 = float(stats[0])  # AP@[.5:.95]
            overall_map50 = float(stats[1])     # AP@.5
        except (IndexError, TypeError, ValueError) as e:
            overall_map50_95 = 0.0
            overall_map50 = 0.0
    else:
        overall_map50_95 = 0.0
        overall_map50 = 0.0
    
    # Bug 2修复：正确计算总体P和R
    # 如果 COCO stats 为 0，尝试从各类别计算平均值
    if overall_map50 == 0.0 and precision is not None and precision.size > 0:
        # 计算所有类别的平均 mAP50
        class_map50_list = []
        for idx in range(precision.shape[2]):  # 遍历所有类别
            p_iou50 = precision[0, :, idx, 0, 2]  # IoU=0.5
            p_iou50 = p_iou50[p_iou50 > -1]
            if len(p_iou50) > 0:
                class_map50_list.append(np.mean(p_iou50))
        
        if len(class_map50_list) > 0:
            overall_map50 = np.mean(class_map50_list)
        
        # 计算所有类别的平均 mAP50-95
        class_map50_95_list = []
        for idx in range(precision.shape[2]):
            p_all_iou = precision[:, :, idx, 0, 2]
            p_all_iou = p_all_iou[p_all_iou > -1]
            if len(p_all_iou) > 0:
                class_map50_95_list.append(np.mean(p_all_iou))
        
        if len(class_map50_95_list) > 0:
            overall_map50_95 = np.mean(class_map50_95_list)
    
    # 计算总体P和R（更接近 Ultralytics 的 max-F1 选点风格）：
    # 1) IoU=0.5 下取每类 precision + score 阈值轨迹
    # 2) 在 confidence 轴(1000点)插值出每类 P/R 曲线
    # 3) 对 mean(F1) 做轻量平滑后取全局最佳点
    # 说明：AP 仍使用 COCO 官方 stats；此处仅用于 YOLO 风格 P/R 展示。
    iou_idx_for_p = 0  # IoU=0.5
    dense_x = np.linspace(0.0, 1.0, 1000)
    recall_thresholds = np.array(getattr(coco_eval.params, 'recThrs', np.linspace(0.0, 1.0, 101)), dtype=float)

    def smooth_curve(y: np.ndarray, frac: float = 0.1) -> np.ndarray:
        if y.size < 3:
            return y
        nf = max(3, round(len(y) * frac * 2) // 2 + 1)  # odd filter size
        pad = np.ones(nf // 2, dtype=float)
        yp = np.concatenate((pad * y[0], y, pad * y[-1]))
        kernel = np.ones(nf, dtype=float) / nf
        return np.convolve(yp, kernel, mode='valid')

    # 缓存每类在全局最佳点处的 P/R，供后续 per-class 打印与写入
    class_pr_values = {}  # key: precision_idx -> (p_value, r_value)
    overall_p = 0.0
    overall_r = 0.0

    if precision is not None and precision.size > 0:
        p_curve_all = []
        r_curve_all = []
        curve_class_indices = []

        for cat_idx in range(precision.shape[2]):  # 遍历 precision 类别维
            p_raw = precision[iou_idx_for_p, :, cat_idx, 0, 2].astype(float).copy()  # IoU=0.5
            s_raw = None
            if scores is not None:
                s_raw = scores[iou_idx_for_p, :, cat_idx, 0, 2].astype(float).copy()

            valid = p_raw > -1
            if s_raw is not None:
                valid = valid & (s_raw > -1)
            if not np.any(valid):
                continue

            r_raw = recall_thresholds[valid]
            p_raw = p_raw[valid]
            if s_raw is not None:
                s_raw = s_raw[valid]
                # 按 confidence 从高到低排序，匹配 YOLO 的按置信度曲线逻辑
                order = np.argsort(s_raw)[::-1]
                conf_desc = s_raw[order]
                p_desc = p_raw[order]
                r_desc = r_raw[order]

                # 去重（保留降序序列中的首个点），便于插值稳定
                keep = np.ones_like(conf_desc, dtype=bool)
                if len(conf_desc) > 1:
                    keep[1:] = conf_desc[1:] != conf_desc[:-1]
                conf_desc = conf_desc[keep]
                p_desc = p_desc[keep]
                r_desc = r_desc[keep]

                p_interp = np.interp(-dense_x, -conf_desc, p_desc, left=1.0, right=p_desc[-1])
                r_interp = np.interp(-dense_x, -conf_desc, r_desc, left=0.0, right=r_desc[-1])
            else:
                # 回退：若缺失 score 曲线，仅用 recall 轴近似
                order = np.argsort(r_raw)
                r_raw = r_raw[order]
                p_raw = p_raw[order]
                if len(r_raw) == 1:
                    p_interp = np.full_like(dense_x, p_raw[0], dtype=float)
                else:
                    p_interp = np.interp(dense_x, r_raw, p_raw, left=p_raw[0], right=p_raw[-1])
                r_interp = dense_x.copy()
            p_curve_all.append(p_interp)
            r_curve_all.append(r_interp)
            curve_class_indices.append(cat_idx)

        if len(p_curve_all) > 0:
            p_curve_all = np.stack(p_curve_all, axis=0)
            r_curve_all = np.stack(r_curve_all, axis=0)
            f1_curve_all = 2 * p_curve_all * r_curve_all / (p_curve_all + r_curve_all + 1e-16)

            best_i = int(np.argmax(smooth_curve(f1_curve_all.mean(0), 0.1)))
            best_p = p_curve_all[:, best_i]
            best_r = r_curve_all[:, best_i]

            for idx_in_curve, cat_idx in enumerate(curve_class_indices):
                class_pr_values[cat_idx] = (float(best_p[idx_in_curve]), float(best_r[idx_in_curve]))

            overall_p = float(np.mean(best_p))
            overall_r = float(np.mean(best_r))
    
    # 先打印 all 行（YOLOv8格式）
    for cat in cats:
        cat_instances = 0
        for img_id in coco_eval.params.imgIds:
            ann_ids = base_ds.getAnnIds(imgIds=img_id, catIds=cat['id'])
            cat_instances += len(ann_ids)
        all_instances += cat_instances
    
    print(f"{'all':>15} {all_images:>8} {all_instances:>10} "
          f"{overall_p:>10.4f} {overall_r:>10.4f} {overall_map50:>10.4f} {overall_map50_95:>12.4f}")
    
    # 打印每个类别的统计
    for idx, cat in enumerate(cats):
        cat_id = cat['id']
        cat_name = cat['name']
        
        # 使用正确的类别索引映射
        if cat_id not in cat_id_to_idx:
            continue
        precision_idx = cat_id_to_idx[cat_id]
        
        # 计算该类别的实例数
        cat_instances = 0
        for img_id in coco_eval.params.imgIds:
            ann_ids = base_ds.getAnnIds(imgIds=img_id, catIds=cat_id)
            cat_instances += len(ann_ids)
        
        if cat_instances == 0:
            continue
        
        # mAP50-95: precision在所有IoU阈值上的平均
        p_all_iou = precision[:, :, precision_idx, 0, 2]  # [10, 101]
        p_all_iou = p_all_iou[p_all_iou > -1]
        map50_95 = np.mean(p_all_iou) if len(p_all_iou) > 0 else 0
        
        # mAP50: precision在IoU=0.5时的平均
        p_iou50 = precision[0, :, precision_idx, 0, 2]  # [101]
        p_iou50 = p_iou50[p_iou50 > -1]
        map50 = np.mean(p_iou50) if len(p_iou50) > 0 else 0
        
        # Precision/Recall：使用与 overall 同一全局 best-F1 点（YOLO-style）
        p_value, r_value = class_pr_values.get(precision_idx, (0.0, 0.0))
        
        # 打印该类别的统计（YOLOv8格式）
        print(f"{cat_name:>15} {all_images:>8} {cat_instances:>10} "
              f"{p_value:>10.4f} {r_value:>10.4f} {map50:>10.4f} {map50_95:>12.4f}")
        
        # 记录每个类别的指标到字典
        category_stats[f'metrics/{cat_name}_P'] = float(p_value)
        category_stats[f'metrics/{cat_name}_R'] = float(r_value)
        category_stats[f'metrics/{cat_name}_mAP50'] = float(map50)
        category_stats[f'metrics/{cat_name}_mAP50-95'] = float(map50_95)
    
    # 添加总体Precision和Recall到返回值
    category_stats['overall_precision'] = float(overall_p)
    category_stats['overall_recall'] = float(overall_r)
    
    return category_stats
