# 数据工程 Benchmark 使用指南

## 目标

在**模型完全不变**的前提下，通过优化数据处理来提升 `final_score`。

评估指标: `final_score = (pred_return - random_return) / (max_return - random_return)`
- 1.0 = 完美预测 Top5
- 0.0 = 随机策略
- 越高越好

## 快速开始

```bash
# 1. 安装环境（需要先安装 uv: pip install uv）
uv sync

# 2. 安装 TA-Lib（技术指标库，需要预编译 wheel）
#    Windows: pip install TA-Lib‑0.4.32‑cp312‑cp312‑win_amd64.whl
#    或者: conda install -c conda-forge ta-lib

# 3. 运行 benchmark（包含 baseline 对比）
uv run python data_benchmark.py
```

## 你需要做什么

1. 打开 `data_benchmark.py`
2. 找到标记 `▶▶▶ 在下方实现你的 DataProcessor 子类 ◀◀◀` 的区域
3. 实现你的处理器类：

```python
class MyProcessor(DataProcessor):
    def name(self) -> str:
        return "MyProcessor"

    def process(self, df: pd.DataFrame, stockid2idx: dict) -> tuple:
        df = df.copy()
        df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

        # ===== 你的特征工程 =====
        # 例: df['my_feature'] = df['收盘'].rolling(10).mean()

        # 映射股票索引（必须）
        df['instrument'] = df['股票代码'].map(stockid2idx)
        df = df.dropna(subset=['instrument']).copy()
        df['instrument'] = df['instrument'].astype(np.int64)

        # 构建标签（必须调用）
        df = _build_label_and_clean(df, drop_small_open=True)

        # 定义特征列
        feature_columns = ['my_feature', ...]

        return df, feature_columns
```

4. 把你的处理器注册到 `PROCESSORS` 列表：
```python
PROCESSORS = [
    Baseline39(),
    Baseline158plus39(),
    MyProcessor(),  # <-- 加这里
]
```

5. 运行 `python data_benchmark.py`

## 你可以修改的

| 维度 | 可以做 |
|------|--------|
| **特征工程** | 任意新特征、特征选择、特征变换 |
| **数据清洗** | 异常值处理、缺失值策略、去极值 |
| **数据源** | 使用 Tushare 数据（见 `Data_managemengt/`）|
| **标准化** | 换用其他 scaler、分位数变换等 |
| **股票筛选** | 去掉特定股票、按流动性筛选 |
| **时间范围** | 调整训练/验证时间窗口 |

## 你不可以修改的

| 维度 | 原因 |
|------|------|
| 模型架构 (`model.py`) | 控制变量 |
| 超参数 (`config.py`) | 控制变量 |
| 损失函数 (`WeightedRankingLoss`) | 控制变量 |
| 评估指标 (`calculate_ranking_metrics`) | 评估标准统一 |
| 训练循环 (`train_ranking_model`) | 控制变量 |

## 输出文件

```
benchmark_runs/
  {timestamp}/
    {YourProcessorName}/
      best_model.pth      # 模型权重
      scaler.pkl           # 标准化器
      result.json          # 结果指标
    summary.json           # 所有处理器的汇总对比
```

## 输入数据格式

`data/train.csv` 的列:

| 列名 | 说明 |
|------|------|
| 股票代码 | 如 sh.600000 |
| 日期 | 如 2024-01-02 |
| 开盘 | 开盘价 |
| 收盘 | 收盘价 |
| 最高 | 最高价 |
| 最低 | 最低价 |
| 成交量 | 成交量 |
| 成交额 | 成交额 |
| 振幅 | 振幅(%) |
| 涨跌额 | 涨跌额 |
| 换手率 | 换手率(%) |
| 涨跌幅 | 涨跌幅(%) |

## 外部数据

`Data_managemengt/data_tushare/` 目录下有更丰富的数据:

- `stock_data.csv` — 基础 OHLCV (12列)
- `stock_data_tushare_tradable.csv` — 扩展特征 (92列，含估值/资金流/两融)
- `stock_data_tushare_wide.csv` — 全 A 股宽表
- `a_stock_data.db` — SQLite 原始数据库

你可以把这些数据融合进来，只要最终 DataFrame 格式兼容即可。

## Tips

1. **先跑 baseline**，了解基准分数
2. **特征不在多而在精**——噪声特征会拖累模型
3. **数据质量 > 特征数量**——缺失值、异常值处理很关键
4. **注意过拟合**——训练集涨但验证集不涨说明过拟合了
5. **时间序列特性**——不要用未来信息泄露到特征中
