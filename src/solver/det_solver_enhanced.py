'''
Enhanced Detection Solver with YOLO-style logging and early stopping
by lyuwenyu (modified)
'''
import time 
import json
import datetime
import csv
from pathlib import Path

import torch 
import torch.nn as nn

from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine_enhanced import train_one_epoch, evaluate


class DetSolverEnhanced(BaseSolver):
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.best_map = 0.0
        self.best_map50 = 0.0
        self.best_epoch = -1
        self.patience_counter = 0
        # 从 yaml_cfg 或 cfg 属性读取早停配置
        self.early_stopping_patience = cfg.yaml_cfg.get('early_stopping_patience', getattr(cfg, 'early_stopping_patience', 300))
        self.val_interval = cfg.yaml_cfg.get('val_interval', 1)  # 每隔N轮验证一次，默认每轮
        self.results_file = None

    def state_dict(self, last_epoch=None):
        state = super().state_dict(last_epoch)
        state['solver_state'] = {
            'best_map': self.best_map,
            'best_map50': self.best_map50,
            'best_epoch': self.best_epoch,
            'patience_counter': self.patience_counter,
        }
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)

        solver_state = state.get('solver_state', {})
        self.best_map = float(solver_state.get('best_map', self.best_map))
        self.best_map50 = float(solver_state.get('best_map50', self.best_map50))
        self.best_epoch = int(solver_state.get('best_epoch', self.best_epoch))
        self.patience_counter = int(solver_state.get('patience_counter', self.patience_counter))
        
    def print_model_info(self):
        """任务4: 训练前自检输出 - 打印模型架构和超参数"""
        if not dist.is_main_process():
            return
        
        # 打印分隔线
        print("\n")
        print("=" * 100)
        print(" " * 35 + "MODEL ARCHITECTURE")
        print("=" * 100)
        
        # 打印完整模型结构
        print(self.model)
        
        # 先计算一次“真实”的参数统计（避免第三方 summary 可能的重复计数/显示不一致）
        true_total_params = sum(p.numel() for p in self.model.parameters())
        true_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        true_non_trainable_params = true_total_params - true_trainable_params
        true_params_size_mb = true_total_params * 4 / 1024 / 1024

        # 使用 torchinfo 打印详细的模型摘要（对输出做一次对齐，避免出现 Total params 与 Params size 不一致）
        try:
            import re
            import io
            from contextlib import redirect_stdout
            from torchinfo import summary
            print("\n" + "=" * 100)
            print(" " * 35 + "MODEL SUMMARY (torchinfo)")
            print("=" * 100)

            buf = io.StringIO()
            with redirect_stdout(buf):
                summary(
                    self.model,
                    input_size=(1, 3, 640, 640),
                    col_names=["output_size", "num_params"],
                    depth=5,
                    verbose=1,
                )
            out = buf.getvalue()

            # 尝试解析 torchinfo 的 Input/Fwd&Bwd 大小，便于同步 Estimated Total Size
            def _find_float(pattern: str):
                m = re.search(pattern, out)
                return float(m.group(1)) if m else None

            input_mb = _find_float(r"Input size \(MB\):\s*([0-9.]+)")
            fwbw_mb = _find_float(r"Forward/backward pass size \(MB\):\s*([0-9.]+)")
            est_mb = None
            if input_mb is not None and fwbw_mb is not None:
                est_mb = input_mb + fwbw_mb + true_params_size_mb

            # 用“真实”统计替换 torchinfo 末尾汇总（不同 torchinfo 版本/共享参数会导致汇总行不一致）
            out = re.sub(r"Total params:\s*[0-9,]+", f"Total params: {true_total_params:,}", out)
            out = re.sub(r"Trainable params:\s*[0-9,]+", f"Trainable params: {true_trainable_params:,}", out)
            out = re.sub(r"Non-trainable params:\s*[0-9,]+", f"Non-trainable params: {true_non_trainable_params:,}", out)
            out = re.sub(r"Params size \(MB\):\s*[0-9.]+", f"Params size (MB): {true_params_size_mb:.2f}", out)
            if est_mb is not None:
                out = re.sub(r"Estimated Total Size \(MB\):\s*[0-9.]+", f"Estimated Total Size (MB): {est_mb:.2f}", out)

            print(out, end="")
        except ImportError:
            print("\nWarning: torchinfo not installed. Run 'pip install torchinfo' to see detailed summary.")
        except Exception as e:
            print(f"\nWarning: torchinfo summary failed ({type(e).__name__}: {e}). Continue without torchinfo.")
        
        # 打印模型统计
        print("\n" + "-" * 100)
        print("MODEL STATISTICS")
        print("-" * 100)
        
        n_parameters = true_total_params
        n_trainable = true_trainable_params
        
        print(f"{'Total Parameters':30s}: {n_parameters:,}")
        print(f"{'Trainable Parameters':30s}: {n_trainable:,}")
        print(f"{'Non-trainable Parameters':30s}: {n_parameters - n_trainable:,}")
        print(f"{'Model Size (MB)':30s}: {true_params_size_mb:.2f}")
        
        # 打印超参数
        print("\n" + "=" * 100)
        print(" " * 35 + "HYPERPARAMETERS")
        print("=" * 100)
        
        cfg = self.cfg
        
        # 训练配置
        print("\n[Training Configuration]")
        print("-" * 50)
        print(f"{'Task':30s}: {cfg.yaml_cfg.get('task', 'detection')}")
        print(f"{'Total Epochs':30s}: {cfg.epoches}")
        print(f"{'Early Stopping Patience':30s}: {self.early_stopping_patience}")
        print(f"{'Clip Max Norm':30s}: {cfg.clip_max_norm}")
        print(f"{'Use AMP':30s}: {cfg.use_amp}")
        print(f"{'Use EMA':30s}: {cfg.use_ema}")
        
        # 优化器配置
        print("\n[Optimizer Configuration]")
        print("-" * 50)
        opt_cfg = cfg.yaml_cfg.get('optimizer', {})
        print(f"{'Optimizer Type':30s}: {opt_cfg.get('type', 'N/A')}")
        print(f"{'Base Learning Rate':30s}: {opt_cfg.get('lr', 'N/A')}")
        # 显示配置文件中的 weight_decay（默认值）
        print(f"{'Weight Decay (default)':30s}: {opt_cfg.get('weight_decay', 'N/A')}")
        print(f"{'Betas':30s}: {opt_cfg.get('betas', 'N/A')}")
        # 显示参数组数量和各组的 weight_decay
        if hasattr(self, 'optimizer'):
            print(f"{'Param Groups':30s}: {len(self.optimizer.param_groups)}")
            # 显示各组的 weight_decay（详细）
            for i, pg in enumerate(self.optimizer.param_groups):
                wd = pg.get('weight_decay', 0)
                lr = pg.get('lr', 0)
                n_params = len(list(pg['params']))
                print(f"  Group {i}: {n_params} params, lr={lr:.6f}, wd={wd}")
        
        # 学习率调度器配置
        print("\n[LR Scheduler Configuration]")
        print("-" * 50)
        lr_cfg = cfg.yaml_cfg.get('lr_scheduler', {})
        print(f"{'Scheduler Type':30s}: {lr_cfg.get('type', 'N/A')}")
        print(f"{'T_max':30s}: {lr_cfg.get('T_max', 'N/A')}")
        print(f"{'Warmup Epochs':30s}: {lr_cfg.get('warmup_epochs', 'N/A')}")
        print(f"{'Eta Min':30s}: {lr_cfg.get('eta_min', 'N/A')}")
        
        # 数据配置
        print("\n[Data Configuration]")
        print("-" * 50)
        print(f"{'Num Classes':30s}: {cfg.yaml_cfg.get('num_classes', 'N/A')}")
        if 'ordinal_num_classes' in cfg.yaml_cfg:
            print(f"{'Ordinal Classes':30s}: {cfg.yaml_cfg.get('ordinal_num_classes')}")
        print(f"{'Train Batch Size':30s}: {cfg.yaml_cfg.get('train_dataloader', {}).get('batch_size', 'N/A')}")
        print(f"{'Val Batch Size':30s}: {cfg.yaml_cfg.get('val_dataloader', {}).get('batch_size', 'N/A')}")
        print(f"{'Input Size':30s}: 640 x 640")
        
        print("\n" + "=" * 100 + "\n")
    
    def print_dataset_info(self):
        """打印数据集信息"""
        if not dist.is_main_process():
            return
        
        print("=" * 100)
        print(" " * 35 + "DATASET INFO")
        print("=" * 100)
        
        # 训练集信息
        train_dataset = self.train_dataloader.dataset
        val_dataset = self.val_dataloader.dataset
        
        print(f"\n{'Train Dataset':30s}: {len(train_dataset)} images")
        print(f"{'Val Dataset':30s}: {len(val_dataset)} images")
        print(f"{'Train Batches per Epoch':30s}: {len(self.train_dataloader)}")
        print(f"{'Val Batches per Epoch':30s}: {len(self.val_dataloader)}")
        
        # 计算warmup步数
        warmup_epochs = self.cfg.yaml_cfg.get('lr_scheduler', {}).get('warmup_epochs', 0)
        warmup_steps = warmup_epochs * len(self.train_dataloader)
        print(f"{'Warmup Steps':30s}: {warmup_steps} ({warmup_epochs} epochs)")
        
        print("\n" + "=" * 100 + "\n")
    
    def init_results_csv(self):
        """任务8: 初始化results.csv文件（延迟写入表头，等待第一次update时获取类别信息）"""
        if not dist.is_main_process():
            return
            
        self.results_file = self.output_dir / 'results.csv'
        self.csv_headers_written = False
        self.category_names = []  # 存储类别名称列表
        self._csv_loss_keys = []

        if self.cfg.resume and self.results_file.exists():
            with open(self.results_file, newline='') as f:
                reader = csv.reader(f)
                headers = next(reader, [])

            if headers:
                self.csv_headers_written = True
                self._csv_loss_keys = [
                    col.split('/', 1)[1]
                    for col in headers
                    if col.startswith('train/loss_') and '_aux' not in col
                ]
    
    
    def update_results_csv(self, epoch, train_stats, test_stats, lr):
        """Write one epoch of stats to results.csv.
        - Loss columns are inferred dynamically from train_stats/test_stats (only losses that exist will be written).
        - Always writes total loss (train/total_loss, val/total_loss).
        """
        # --- mAP ---
        map50 = test_stats.get('coco_eval_bbox_mAP50', 0)
        map50_95 = test_stats.get('coco_eval_bbox_mAP', 0)

        # --- total loss (criterion already applies weights; stats['loss'] is the total) ---
        train_total = train_stats.get('loss', 0.0)
        val_total = test_stats.get('loss', 0.0)

        # --- infer active loss keys (base only) ---
        def _is_base_loss_key(k: str) -> bool:
            if not isinstance(k, str):
                return False
            if not k.startswith('loss_'):
                return False
            if '_aux' in k:
                return False
            if 'accuracy' in k or 'cardinality' in k:
                return False
            return True

        loss_keys = sorted({k for k in list(train_stats.keys()) + list(test_stats.keys()) if _is_base_loss_key(k)})
        preferred = ['loss_vfl', 'loss_ordinal', 'loss_ord_branch', 'loss_bbox', 'loss_giou', 'loss_dfl']
        loss_keys = [k for k in preferred if k in loss_keys] + [k for k in loss_keys if k not in preferred]

        # cache for consistent header/order across epochs in a run
        if not hasattr(self, '_csv_loss_keys') or not self._csv_loss_keys:
            self._csv_loss_keys = loss_keys
        else:
            # keep original order, but allow newly appeared losses (rare) to be appended
            for k in loss_keys:
                if k not in self._csv_loss_keys:
                    self._csv_loss_keys.append(k)
        loss_keys = self._csv_loss_keys

        # --- per-class metrics ---
        category_metrics = self.per_class_metrics.get(epoch, {}) if hasattr(self, 'per_class_metrics') else {}

        # --- write header once ---
        if not getattr(self, 'csv_headers_written', False):
            headers = ['epoch', 'train/total_loss', 'val/total_loss']
            headers += [f'train/{k}' for k in loss_keys]
            headers += [f'val/{k}' for k in loss_keys]
            headers += ['mAP50', 'mAP50-95', 'lr']
            for cat_name in self.category_names:
                headers.extend([
                    f'metrics/{cat_name}_P',
                    f'metrics/{cat_name}_R',
                    f'metrics/{cat_name}_mAP50',
                    f'metrics/{cat_name}_mAP50-95'
                ])
            with open(self.results_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            self.csv_headers_written = True

        # --- build row ---
        row = [
            str(epoch),
            f'{float(train_total):.6f}',
            f'{float(val_total):.6f}',
        ]
        for k in loss_keys:
            row.append(f'{float(train_stats.get(k, 0.0)):.6f}')
        for k in loss_keys:
            row.append(f'{float(test_stats.get(k, 0.0)):.6f}')

        row += [f'{float(map50):.6f}', f'{float(map50_95):.6f}', f'{float(lr):.8f}']

        for cat_name in self.category_names:
            row.append(f'{float(category_metrics.get(f"metrics/{cat_name}_P", 0.0)):.6f}')
            row.append(f'{float(category_metrics.get(f"metrics/{cat_name}_R", 0.0)):.6f}')
            row.append(f'{float(category_metrics.get(f"metrics/{cat_name}_mAP50", 0.0)):.6f}')
            row.append(f'{float(category_metrics.get(f"metrics/{cat_name}_mAP50-95", 0.0)):.6f}')

        with open(self.results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def fit(self):
        """主训练循环"""
        print("Start training")
        self.train()

        args = self.cfg 
        
        # 任务4: 打印模型信息和超参数（在train()之后，因为model在train()中初始化）
        self.print_model_info()
        
        # 打印数据集信息
        self.print_dataset_info()
        
        # 任务8: 初始化results.csv
        self.init_results_csv()
        
        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)

        # 初始化 per_class_metrics 字典
        self.per_class_metrics = {}
        
        # 从数据集获取类别名称
        if hasattr(base_ds, 'cats') and base_ds.cats:
            self.category_names = [base_ds.cats[cat_id]['name'] for cat_id in sorted(base_ds.cats.keys())]
        else:
            # 使用默认类别名称
            num_classes = self.cfg.yaml_cfg.get('num_classes', getattr(self.cfg, 'num_classes', 3))
            self.category_names = [f'class_{i}' for i in range(num_classes)]
        
        print(f"Category names: {self.category_names}")

        # 训练时间统计
        training_start_time = time.time()
        epoch_times = []
        
        # 学习率记录（用于绘制学习率曲线）
        lr_history = []
        lr_backbone_history = []
        
        for epoch in range(self.last_epoch + 1, args.epoches):
            epoch_start_time = time.time()
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            # 训练一个epoch
            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, 
                self.device, epoch, args.clip_max_norm, 
                print_freq=args.log_step, ema=self.ema, scaler=self.scaler,
                total_epochs=args.epoches
            )

            self.lr_scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # 记录学习率（包括backbone分组）
            lr_history.append(current_lr)
            # 查找backbone分组的学习率
            backbone_lr = current_lr
            for pg in self.optimizer.param_groups:
                if 'backbone' in str(pg.get('params', '')):
                    backbone_lr = pg['lr']
                    break
            lr_backbone_history.append(backbone_lr)
            
            # 跳跃验证：每 val_interval 轮或最后一轮才评估
            should_eval = (epoch % self.val_interval == 0) or (epoch == args.epoches - 1)
            
            if should_eval:
                # 验证
                module = self.ema.module if self.ema else self.model
                test_stats, coco_evaluator = evaluate(
                    module, self.criterion, self.postprocessor, self.val_dataloader, 
                    base_ds, self.device, self.output_dir, epoch, args.epoches
                )
                
                # 直接从 coco_evaluator 获取 mAP
                current_map = 0.0
                current_map50 = 0.0
                
                if coco_evaluator is not None and 'bbox' in coco_evaluator.coco_eval:
                    coco_eval = coco_evaluator.coco_eval['bbox']
                    coco_stats = coco_eval.stats
                    if coco_stats is not None and hasattr(coco_stats, '__len__') and len(coco_stats) > 1:
                        current_map = float(coco_stats[0])
                        current_map50 = float(coco_stats[1])
                
                # 直接复用 evaluate() 已计算好的每类指标（避免重复计算）
                per_class_stats = {k: v for k, v in test_stats.items() if k.startswith('metrics/')}
                self.per_class_metrics[epoch] = per_class_stats
                test_stats['coco_eval_bbox_mAP'] = current_map
                test_stats['coco_eval_bbox_mAP50'] = current_map50
                self.update_results_csv(epoch, train_stats, test_stats, current_lr)
                
                # 保存best.pt
                if self.output_dir and dist.is_main_process():
                    if current_map > self.best_map or self.best_epoch == -1:
                        self.best_map = current_map
                        self.best_map50 = current_map50
                        self.best_epoch = epoch
                        best_path = self.output_dir / 'best.pt'
                        dist.save_on_master(self.state_dict(epoch), best_path)
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                
                if dist.is_main_process():
                    print(f"best_stat: epoch={self.best_epoch + 1}, mAP50={self.best_map50:.4f}, mAP50-95={self.best_map:.4f}")
                    train_loss = train_stats.get('loss', 0)
                    train_bbox = train_stats.get('loss_bbox', 0)
                    train_vfl = train_stats.get('loss_vfl', train_stats.get('loss_ordinal', 0))
                    train_giou = train_stats.get('loss_giou', 0)
                    val_loss = test_stats.get('loss', 0)
                    val_bbox = test_stats.get('loss_bbox', 0)
                    val_vfl = test_stats.get('loss_vfl', test_stats.get('loss_ordinal', 0))
                    val_giou = test_stats.get('loss_giou', 0)
                    print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, train_cls={train_vfl:.4f}, "
                          f"train_bbox={train_bbox:.4f}, train_giou={train_giou:.4f}; "
                          f"val_loss={val_loss:.4f}, val_cls={val_vfl:.4f}, "
                          f"val_bbox={val_bbox:.4f}, val_giou={val_giou:.4f}\n")
                
                if self.patience_counter >= self.early_stopping_patience:
                    print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                    print(f'Best mAP50-95: {self.best_map:.4f} at epoch {self.best_epoch}')
                    break
            else:
                # 非评估轮：只打印训练统计
                test_stats = {}
                if dist.is_main_process():
                    train_loss = train_stats.get('loss', 0)
                    print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f} [skip eval, next @ epoch {(epoch // self.val_interval + 1) * self.val_interval}]\n")
            
            # 保存last.pt (每轮)
            if self.output_dir and dist.is_main_process():
                last_path = self.output_dir / 'last.pt'
                dist.save_on_master(self.state_dict(epoch), last_path)

            # 保存日志
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'lr': current_lr
            }

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - training_start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'\nTraining completed in {total_time_str}')
        print(f'Best mAP50-95: {self.best_map:.4f} at epoch {self.best_epoch}')
        
        # 绘制学习率曲线
        if dist.is_main_process() and lr_history:
            self.plot_lr_curve(lr_history, lr_backbone_history)
        
        # 任务9: 训练后自动测试
        self.auto_test_on_best()

    def plot_lr_curve(self, lr_history, lr_backbone_history):
        """绘制学习率曲线，验证 Warmup + Cosine 调度是否正确"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        epochs = list(range(len(lr_history)))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle('Learning Rate Schedule (Warmup + Cosine Annealing)', fontsize=14, fontweight='bold')
        
        # 绘制主学习率
        ax.plot(epochs, lr_history, 'b-', linewidth=2, label='Global LR', marker='o', markersize=3)
        
        # 如果backbone学习率不同，也绘制出来
        if lr_backbone_history and lr_backbone_history != lr_history:
            ax.plot(epochs, lr_backbone_history, 'r--', linewidth=2, label='Backbone LR', marker='s', markersize=3)
        
        # 标注关键点
        if len(lr_history) > 0:
            # 标注初始学习率
            ax.annotate(f'Start: {lr_history[0]:.6f}', 
                       xy=(0, lr_history[0]), xytext=(5, 10), textcoords='offset points',
                       fontsize=9, color='blue')
            
            # 标注最大学习率（warmup结束后）
            max_lr = max(lr_history)
            max_idx = lr_history.index(max_lr)
            ax.annotate(f'Max: {max_lr:.6f} (ep{max_idx})', 
                       xy=(max_idx, max_lr), xytext=(5, 10), textcoords='offset points',
                       fontsize=9, color='green')
            
            # 标注最终学习率
            ax.annotate(f'End: {lr_history[-1]:.6f}', 
                       xy=(len(lr_history)-1, lr_history[-1]), xytext=(-50, 10), textcoords='offset points',
                       fontsize=9, color='red')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_yscale('log')  # 使用对数刻度更清晰
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # 添加warmup区域标注
        warmup_epochs = getattr(self.cfg, 'warmup_epochs', 5)
        if warmup_epochs < len(lr_history):
            ax.axvline(x=warmup_epochs, color='gray', linestyle=':', alpha=0.7)
            ax.text(warmup_epochs + 0.5, max(lr_history) * 0.5, f'Warmup ends\n(ep{warmup_epochs})', 
                   fontsize=8, color='gray')
        
        plt.tight_layout()
        save_path = self.output_dir / 'actual_lr_curve.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Learning rate curve saved to {save_path}")

    def auto_test_on_best(self):
        """任务9: 训练结束后自动在测试集上评估best.pt"""
        if not dist.is_main_process():
            return
            
        print("\n" + "="*80)
        print("AUTOMATIC TESTING ON TEST SET WITH BEST WEIGHTS")
        print("="*80)
        
        best_path = self.output_dir / 'best.pt'
        if not best_path.exists():
            print(f"Warning: best.pt not found at {best_path}")
            return
        
        # 加载best权重
        print(f"Loading best weights from {best_path}")
        state = torch.load(best_path, map_location='cpu')
        base_model = dist.de_parallel(self.model)
        if 'model' in state:
            base_model.load_state_dict(state['model'])
        else:
            base_model.load_state_dict(state)

        module = base_model
        if self.ema is not None and 'ema' in state:
            self.ema.load_state_dict(state['ema'])
            module = self.ema.module
        
        if 'test_dataloader' not in self.cfg.yaml_cfg:
            print("Warning: test_dataloader not configured, using val_dataloader instead")
            test_dataloader = self.val_dataloader
            test_dataset = self.val_dataloader.dataset
        else:
            print("Creating test dataloader from YAML test_dataloader")
            test_dataloader = self.cfg.test_dataloader
            test_dataset = test_dataloader.dataset
        
        base_ds = get_coco_api_from_dataset(test_dataset)
        
        # 在测试集上评估
        test_stats, coco_evaluator = evaluate(
            module, self.criterion, self.postprocessor, test_dataloader,
            base_ds, self.device, self.output_dir, epoch=-1, total_epochs=-1, 
            is_test=True
        )
        
        print("\n" + "="*80)
        print("TEST SET EVALUATION COMPLETED")
        print("="*80 + "\n")

    def val(self):
        """验证模式"""
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module, self.criterion, self.postprocessor,
            self.val_dataloader, base_ds, self.device, self.output_dir,
            epoch=0, total_epochs=1
        )
                
        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return
