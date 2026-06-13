"""
=============================================================================
数据工程 Benchmark 脚本 — 考验数据处理能力
=============================================================================
用途：
  - 模型架构、超参数、训练流程、损失函数、评估指标全部冻结不可修改
  - 唯一可修改的是数据处理流水线（DataProcessor 类）
  - 数据工程师需要实现自己的 DataProcessor 子类来提升 final_score

使用方法：
  1. 在本文件底部的 YOUR PROCESSOR 区域实现你的 DataProcessor 子类
  2. 运行: python data_benchmark.py
  3. 查看输出的 final_score，与 baseline 对比

评估指标: final_score = (pred_return - random_return) / (max_return - random_return)
  - 理论最高值 1.0（完美预测 Top5）
  - 随机策略约 0.0
  - 越高越好

运行要求：
  使用项目虚拟环境: .venv/Scripts/python.exe data_benchmark.py
=============================================================================
"""

import sys
import os

# 检查关键依赖是否可用
try:
    import torch
except ImportError:
    print("[错误] 请使用项目虚拟环境运行此脚本:")
    print("  .venv\\Scripts\\python.exe data_benchmark.py")
    sys.exit(1)
import json
import time
import hashlib
import multiprocessing as mp
from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib

# ============================================================================
# 0. 导入冻结的模型组件（不可修改）
# ============================================================================
# 将 code/src 加入 path，以便导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code', 'src'))

from config import config as frozen_config
from model import StockTransformer
from train import (
    WeightedRankingLoss,
    RankingDataset,
    collate_fn,
    train_ranking_model,
    evaluate_ranking_model,
    calculate_ranking_metrics,
    split_train_val_by_last_month,
    set_seed,
    feature_cloums_map,
    feature_engineer_func_map,
)

# ============================================================================
# 1. 配置区（冻结，不可修改）
# ============================================================================
BENCHMARK_CONFIG = {
    'data_path': frozen_config['data_path'],
    'sequence_length': frozen_config['sequence_length'],   # 60
    'd_model': frozen_config['d_model'],                   # 256
    'nhead': frozen_config['nhead'],                       # 4
    'num_layers': frozen_config['num_layers'],             # 3
    'dim_feedforward': frozen_config['dim_feedforward'],   # 512
    'batch_size': frozen_config['batch_size'],             # 4
    'num_epochs': frozen_config['num_epochs'],             # 50
    'learning_rate': frozen_config['learning_rate'],       # 1e-5
    'dropout': frozen_config['dropout'],                   # 0.1
    'max_grad_norm': frozen_config['max_grad_norm'],       # 5.0
    'pairwise_weight': frozen_config['pairwise_weight'],   # 1
    'base_weight': frozen_config['base_weight'],           # 1.0
    'top5_weight': frozen_config['top5_weight'],           # 2.0
    'seed': 42,
    'val_months': 2,        # 验证集取最后几个月
    'min_stocks_per_day': 10,  # 每日最少股票数
}

# 输出目录按时间戳隔离，避免覆盖
RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
BENCHMARK_OUTPUT = f'./benchmark_runs/{RUN_ID}'

# ============================================================================
# 2. DataProcessor 基类（数据工程师需要实现的接口）
# ============================================================================
class DataProcessor(ABC):
    """
    数据处理器基类。

    你需要实现 process() 方法，它接收原始 DataFrame，返回处理后的 DataFrame。

    输入 DataFrame 列:
        股票代码, 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌额, 换手率, 涨跌幅

    输出 DataFrame 必须包含:
        - 原始列保持不变（股票代码, 日期, 开盘 等）
        - 'instrument' 列（int64, 股票代码的数值映射）
        - 特征列（你定义的特征名，可以是任意列名）
        - 'label' 列（5日前瞻收益率，已为你计算好，不要修改）

    注意事项:
        - 输出的 DataFrame 中每只股票至少需要 60 行（sequence_length）
        - 你可以自由添加特征、处理缺失值、去除异常值
        - 你可以自由决定使用哪些股票、哪些日期范围
        - 不要删除 'label' 列或修改它的计算方式
    """

    @abstractmethod
    def name(self) -> str:
        """返回处理器名称，用于日志和报告"""
        pass

    @abstractmethod
    def process(self, df: pd.DataFrame, stockid2idx: dict) -> tuple:
        """
        处理数据并返回 (processed_df, feature_columns)

        Args:
            df: 原始 DataFrame，包含股票数据
            stockid2idx: 股票代码到索引的映射 dict

        Returns:
            (processed_df, feature_columns)
            - processed_df: 处理后的 DataFrame
            - feature_columns: 特征列名列表 (list of str)
        """
        pass


# ============================================================================
# 3. 冻结的工具函数（不可修改）
# ============================================================================

def _build_label_and_clean(processed, drop_small_open=True):
    """构建标签并清洗无效样本（冻结逻辑）"""
    processed = processed.copy()
    processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)
    processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)

    if drop_small_open:
        processed = processed[processed['open_t1'] > 1e-4]

    processed['label'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    processed = processed.dropna(subset=['label'])
    processed.drop(columns=['open_t1', 'open_t5'], inplace=True)
    return processed


def run_benchmark(processor: DataProcessor, num_epochs: int = None):
    """
    运行 benchmark 的主函数。
    给定一个 DataProcessor，完成完整的训练+验证流程。

    Args:
        processor: DataProcessor 实例
        num_epochs: 训练轮数（None 则使用配置值）

    Returns:
        dict: {
            'processor_name': str,
            'best_score': float,
            'best_epoch': int,
            'feature_count': int,
            'train_samples': int,
            'val_samples': int,
            'num_stocks': int,
            'train_time_sec': float,
            'run_id': str,
        }
    """
    cfg = BENCHMARK_CONFIG.copy()
    if num_epochs is not None:
        cfg['num_epochs'] = num_epochs

    run_dir = os.path.join(BENCHMARK_OUTPUT, processor.name().replace(' ', '_'))
    os.makedirs(run_dir, exist_ok=True)

    set_seed(cfg['seed'])

    # ---- 设备选择 ----
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"[设备] {device}")

    # ---- 加载原始数据 ----
    data_file = os.path.join(cfg['data_path'], 'train.csv')
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"找不到训练数据: {data_file}")

    full_df = pd.read_csv(data_file)
    print(f"[数据] 加载 {len(full_df)} 行，{full_df['股票代码'].nunique()} 只股票")

    # ---- 建立股票映射 ----
    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    num_stocks = len(stockid2idx)

    # ---- 划分训练/验证集 ----
    train_df, val_df, val_start = split_train_val_by_last_month(full_df, cfg['sequence_length'])

    # ---- 调用数据处理器 ----
    print(f"\n{'='*60}")
    print(f"[处理器] {processor.name()}")
    print(f"{'='*60}")

    t0 = time.time()

    # 处理训练集
    print("[训练集] 开始处理...")
    train_data, features = processor.process(train_df, stockid2idx)
    print(f"[训练集] 处理完成: {len(train_data)} 行, {len(features)} 个特征")

    # 处理验证集
    print("[验证集] 开始处理...")
    val_data, val_features = processor.process(val_df, stockid2idx)
    # 确保验证集使用与训练集相同的特征列
    common_features = [f for f in features if f in val_data.columns]
    if len(common_features) < len(features):
        missing = set(features) - set(common_features)
        print(f"[警告] 验证集缺少 {len(missing)} 个特征，将用 0 填充: {missing}")
        for col in missing:
            val_data[col] = 0.0
    features = common_features
    print(f"[验证集] 处理完成: {len(val_data)} 行, 使用 {len(features)} 个共同特征")

    # ---- 标准化 ----
    scaler = StandardScaler()
    train_data[features] = train_data[features].replace([np.inf, -np.inf], np.nan)
    val_data[features] = val_data[features].replace([np.inf, -np.inf], np.nan)
    train_data = train_data.dropna(subset=features)
    val_data = val_data.dropna(subset=features)

    train_data[features] = scaler.fit_transform(train_data[features])
    val_data[features] = scaler.transform(val_data[features])

    # ---- 创建排序数据集 ----
    from utils import create_ranking_dataset_vectorized

    print("[排序数据集] 创建训练集...")
    train_seq, train_tgt, train_rel, train_idx = create_ranking_dataset_vectorized(
        train_data, features, cfg['sequence_length']
    )

    print("[排序数据集] 创建验证集...")
    val_seq, val_tgt, val_rel, val_idx = create_ranking_dataset_vectorized(
        val_data, features, cfg['sequence_length'],
        min_window_end_date=val_start.strftime('%Y-%m-%d')
    )

    if len(train_seq) == 0 or len(val_seq) == 0:
        print("[错误] 数据不足，无法创建排序数据集")
        return None

    train_dataset = RankingDataset(train_seq, train_tgt, train_rel, train_idx)
    val_dataset = RankingDataset(val_seq, val_tgt, val_rel, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False,
                            collate_fn=collate_fn, num_workers=0, pin_memory=False)

    # ---- 模型初始化（冻结架构） ----
    model = StockTransformer(input_dim=len(features), config=cfg, num_stocks=num_stocks)
    model.to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[模型] 参数量: {param_count:,}")

    # ---- 损失和优化器（冻结） ----
    criterion = WeightedRankingLoss(
        k=5, temperature=1.0,
        weight_factor=cfg['top5_weight'],
        pairwise_weight=cfg['pairwise_weight'],
        base_weight=cfg['base_weight']
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.2, total_iters=cfg['num_epochs']
    )

    # ---- 训练循环（冻结） ----
    best_score = -float('inf')
    best_epoch = -1
    t_train_start = time.time()

    for epoch in range(cfg['num_epochs']):
        print(f"\n--- Epoch {epoch+1}/{cfg['num_epochs']} [{processor.name()}] ---")

        train_loss, train_metrics = train_ranking_model(
            model, train_loader, criterion, optimizer, device, epoch, writer=None
        )

        eval_loss, eval_metrics = evaluate_ranking_model(
            model, val_loader, criterion, device, writer=None, epoch=epoch
        )

        scheduler.step()

        current_score = eval_metrics.get('final_score', 0.0)
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {eval_loss:.4f}")
        print(f"  Val final_score: {current_score:.4f} | Val ratio_pred: {eval_metrics.get('ratio_pred', 0):.4f}")

        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pth'))
            joblib.dump(scaler, os.path.join(run_dir, 'scaler.pkl'))
            print(f"  >>> 新的最佳分数: {best_score:.4f}")

    train_time = time.time() - t_train_start

    # ---- 保存结果 ----
    result = {
        'processor_name': processor.name(),
        'best_score': round(best_score, 6),
        'best_epoch': best_epoch,
        'feature_count': len(features),
        'train_samples': len(train_seq),
        'val_samples': len(val_seq),
        'num_stocks': num_stocks,
        'train_time_sec': round(train_time, 1),
        'run_id': RUN_ID,
        'config': cfg,
    }

    with open(os.path.join(run_dir, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"[完成] {processor.name()}")
    print(f"  Best final_score: {best_score:.4f} (epoch {best_epoch})")
    print(f"  特征数: {len(features)} | 训练样本: {len(train_seq)} | 验证样本: {len(val_seq)}")
    print(f"  训练耗时: {train_time:.1f}s")
    print(f"  结果保存: {run_dir}")
    print(f"{'='*60}")

    return result


# ============================================================================
# 4. Baseline 处理器（使用项目默认的 158+39 特征集）
# ============================================================================
class Baseline158plus39(DataProcessor):
    """基线处理器：使用项目原有的 158+39 特征工程"""

    def name(self) -> str:
        return "Baseline_158+39"

    def process(self, df: pd.DataFrame, stockid2idx: dict) -> tuple:
        df = df.copy()
        df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

        feature_num = '158+39'
        feature_engineer = feature_engineer_func_map[feature_num]
        feature_columns = feature_cloums_map[feature_num]

        # 多进程特征工程
        groups = [group for _, group in df.groupby('股票代码', sort=False)]
        num_processes = min(10, mp.cpu_count())
        with mp.Pool(processes=num_processes) as pool:
            processed_list = list(tqdm(pool.imap(feature_engineer, groups),
                                       total=len(groups), desc="Baseline 特征工程"))

        processed = pd.concat(processed_list).reset_index(drop=True)

        # 映射股票索引
        processed['instrument'] = processed['股票代码'].map(stockid2idx)
        processed = processed.dropna(subset=['instrument']).copy()
        processed['instrument'] = processed['instrument'].astype(np.int64)

        # 构建标签
        processed = _build_label_and_clean(processed, drop_small_open=True)

        return processed, feature_columns


# ============================================================================
# 5. 另一个简单 baseline：仅 39 个技术指标
# ============================================================================
class Baseline39(DataProcessor):
    """基线处理器：仅使用 39 个 TA-Lib 技术指标"""

    def name(self) -> str:
        return "Baseline_39"

    def process(self, df: pd.DataFrame, stockid2idx: dict) -> tuple:
        df = df.copy()
        df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

        feature_num = '39'
        feature_engineer = feature_engineer_func_map[feature_num]
        feature_columns = feature_cloums_map[feature_num]

        groups = [group for _, group in df.groupby('股票代码', sort=False)]
        num_processes = min(10, mp.cpu_count())
        with mp.Pool(processes=num_processes) as pool:
            processed_list = list(tqdm(pool.imap(feature_engineer, groups),
                                       total=len(groups), desc="39特征 工程"))

        processed = pd.concat(processed_list).reset_index(drop=True)

        processed['instrument'] = processed['股票代码'].map(stockid2idx)
        processed = processed.dropna(subset=['instrument']).copy()
        processed['instrument'] = processed['instrument'].astype(np.int64)

        processed = _build_label_and_clean(processed, drop_small_open=True)

        return processed, feature_columns


# ============================================================================
# ============================================================================
#  ██   ██  ██████  ██    ██     ██████  ██████   █████   ██████ ███████ ███████
#  ██   ██  ██   ██  ██  ██      ██   ██ ██   ██ ██   ██ ██      ██      ██
#  ███████  ██████    ████       ██████  ██████  ███████ ██      █████   ███████
#  ██   ██  ██   ██    ██        ██   ██ ██   ██ ██   ██ ██      ██           ██
#  ██   ██  ██   ██    ██        ██████  ██   ██ ██   ██  ██████ ███████ ███████
# ============================================================================
# ============================================================================
#
#   ▶▶▶  在下方实现你的 DataProcessor 子类  ◀◀◀
#
#   规则:
#     1. 继承 DataProcessor
#     2. 实现 name() 和 process() 方法
#     3. process() 必须返回 (DataFrame, feature_columns_list)
#     4. 输出 DataFrame 必须包含: 股票代码, 日期, 开盘, instrument, label
#     5. 你可以:
#        - 添加任意新特征
#        - 更换数据源（但最终格式要兼容）
#        - 做任意数据清洗、缺失值处理、异常值处理
#        - 使用外部数据（如 Tushare 数据，见 Data_managemengt/ 目录）
#        - 修改标准化策略
#     6. 你不可以:
#        - 修改模型架构
#        - 修改损失函数
#        - 修改评估指标
#        - 修改超参数
#     7. 完成后，把你的处理器加到下方 PROCESSORS 列表中
#
# ============================================================================

# ---------- 示例：自定义处理器模板 ----------
#
# class MyAwesomeProcessor(DataProcessor):
#     """我精心设计的数据处理器"""
#
#     def name(self) -> str:
#         return "MyAwesome"
#
#     def process(self, df: pd.DataFrame, stockid2idx: dict) -> tuple:
#         df = df.copy()
#         df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
#
#         # ===== 你的特征工程代码 =====
#         # 例: df['my_feature'] = df['收盘'].rolling(10).mean()
#         # ...
#
#         # 映射股票索引（必须）
#         df['instrument'] = df['股票代码'].map(stockid2idx)
#         df = df.dropna(subset=['instrument']).copy()
#         df['instrument'] = df['instrument'].astype(np.int64)
#
#         # 构建标签（必须调用）
#         df = _build_label_and_clean(df, drop_small_open=True)
#
#         # 定义你的特征列（不要包含 '股票代码', '日期', 'instrument', 'label'）
#         feature_columns = [col for col in df.columns
#                          if col not in ['股票代码', '日期', '开盘', '收盘', '最高', '最低',
#                                         '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
#                                         'instrument', 'label']]
#
#         return df, feature_columns
#
# ============================================================================


# ============================================================================
# 6. 注册处理器列表
# ============================================================================
# 把你实现的处理器加到这个列表中！
PROCESSORS = [
    Baseline39(),           # 39 个 TA-Lib 特征
    Baseline158plus39(),    # 158+39 特征
    # MyAwesomeProcessor(),  # <-- 放你的处理器在这里
]

# ============================================================================
# 7. 主入口
# ============================================================================
def main():
    print("=" * 70)
    print("  数据工程 Benchmark — 考验你的数据处理能力")
    print(f"  运行 ID: {RUN_ID}")
    print(f"  输出目录: {BENCHMARK_OUTPUT}")
    print(f"  训练轮数: {BENCHMARK_CONFIG['num_epochs']}")
    print("=" * 70)

    all_results = []

    for processor in PROCESSORS:
        try:
            result = run_benchmark(processor)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"\n[错误] 处理器 {processor.name()} 失败: {e}")
            import traceback
            traceback.print_exc()

    # ---- 汇总报告 ----
    if all_results:
        print("\n" + "=" * 70)
        print("  BENCHMARK 汇总报告")
        print("=" * 70)

        # 按分数排序
        all_results.sort(key=lambda x: x['best_score'], reverse=True)

        print(f"\n{'排名':<4} {'处理器':<25} {'final_score':<14} {'特征数':<8} {'训练样本':<10} {'耗时(s)':<10}")
        print("-" * 75)

        for i, r in enumerate(all_results, 1):
            print(f"{i:<4} {r['processor_name']:<25} {r['best_score']:<14.4f} "
                  f"{r['feature_count']:<8} {r['train_samples']:<10} {r['train_time_sec']:<10.1f}")

        # 保存汇总
        summary_path = os.path.join(BENCHMARK_OUTPUT, 'summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n汇总结果已保存: {summary_path}")

        # 显示 baseline 对比
        baseline_scores = {r['processor_name']: r['best_score'] for r in all_results}
        if 'Baseline_158+39' in baseline_scores:
            base = baseline_scores['Baseline_158+39']
            print(f"\n--- 与 Baseline (158+39) 对比 ---")
            for r in all_results:
                diff = r['best_score'] - base
                sign = '+' if diff >= 0 else ''
                print(f"  {r['processor_name']:<25} {r['best_score']:.4f} ({sign}{diff:.4f})")

    print("\n[完成] Benchmark 运行结束")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
