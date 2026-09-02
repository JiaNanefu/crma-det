"""
Carya Dataset Training Script
使用定制化配置训练RT-DETR模型
"""

# 在导入任何可能使用 matplotlib 的模块之前，设置非交互式后端
# 这可以避免 tkinter 在多进程 DataLoader 中的线程冲突
import matplotlib
matplotlib.use('Agg')

import os 
import sys 
import argparse
import time
import warnings
from pathlib import Path

# 过滤 PyTorch 相关警告 (必须在导入 torch 之前设置)
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cpu.amp.autocast.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*use_reentrant.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*None of the inputs have requires_grad.*")

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS

def main(args):
    '''主训练函数'''
    if args.test_only and not args.resume:
        raise ValueError("--test-only 需要配合 --resume 指定要评估的 checkpoint")

    resume_path = None
    if args.resume:
        resume_path = Path(args.resume).expanduser()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    
    # 初始化分布式训练（如果需要）
    dist.init_distributed()
    
    # 设置随机种子
    if args.seed is not None:
        dist.set_seed(args.seed)


    # 加载配置
    cfg = YAMLConfig(
        args.config,
        resume=args.resume, 
        use_amp=args.amp,
        tuning=args.tuning
    )

    # 统一输出目录命名规则：
    # - yml 里的 output_dir 作为 “base 输出目录”（建议形如 ./output/rtdetr/<cfg_name>）
    # - 实际每次训练输出目录在 base 后追加时间戳，避免覆盖
    if resume_path is not None:
        resume_parent = resume_path.resolve().parent
        output_dir = resume_parent.parent if resume_parent.name == 'weights' else resume_parent
    else:
        base_output_dir = (
            Path(args.output_dir)
            if args.output_dir
            else Path(cfg.output_dir) if cfg.output_dir else (Path("./output/rtdetr") / Path(args.config).stem)
        )
        if args.no_timestamp:
            output_dir = base_output_dir
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = base_output_dir.parent / f"{base_output_dir.name}_{timestamp}"

    # 覆盖配置中的 output_dir
    cfg.output_dir = output_dir

    if dist.is_main_process():
        print(f"Training output will be saved to: {output_dir}")


    # 创建solver
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    # 训练或测试
    if args.test_only:
        solver.val()
    else:
        solver.fit()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train RT-DETR on Carya Dataset')
    default_config = Path(__file__).resolve().parent / 'configs' / 'rtdetr' / 'rtdetr_r18_gate_aifi_ordinal.yml'
    
    # 默认使用Carya配置
    parser.add_argument(
        '--config', '-c', 
        type=str, 
        default=str(default_config),
        help='配置文件路径'
    )
    parser.add_argument(
        '--resume', '-r', 
        type=str, 
        default=None,
        help='恢复训练的checkpoint路径'
    )
    
    parser.add_argument(
        '--tuning', '-t', 
        type=str, 
        default=None,
        help='微调的预训练权重路径'
    )
    
    parser.add_argument(
        '--test-only', 
        action='store_true', 
        default=False,
        help='仅进行测试'
    )
    
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument(
        '--amp',
        dest='amp',
        action='store_true',
        help='使用混合精度训练'
    )
    amp_group.add_argument(
        '--no-amp',
        dest='amp',
        action='store_false',
        help='关闭混合精度训练'
    )
    parser.set_defaults(amp=True)
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='随机种子'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='覆盖配置文件中的 output_dir'
    )
    parser.add_argument(
        '--no-timestamp',
        action='store_true',
        default=False,
        help='不在 output_dir 后追加时间戳，便于批量重训时使用固定目录'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("RT-DETR Training on Carya Dataset")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Resume: {args.resume}")
    print(f"Tuning: {args.tuning}")
    print(f"AMP: {args.amp}")
    print(f"Seed: {args.seed}")
    print(f"Test Only: {args.test_only}")
    print("="*80 + "\n")
    
    main(args)
