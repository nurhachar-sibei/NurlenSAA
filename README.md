# NurlenSAA

本仓库提供一套用于配置、运行、评估资产配置策略的通用回测框架，并配套若干基准策略、日志与报表输出工具。通过 `StrategyRunner` 可以一键完成数据读取、仓位准备、策略与基准初始化、回测执行、绩效评估以及图表绘制的全流程。

## 目录与核心模块

```
NurlenSAA/
├── backtest_engine.py      # 事件驱动的多策略回测引擎
├── benchmark_strategy.py   # 固定/动态权重的基准策略与管理器
├── main.py                 # 封装完整流程的 StrategyRunner
├── example/                # HRP 示例策略、配置与输出目录
│   ├── HRP_config.py       # HRP 与基准的参数配置
│   ├── Strategy_HRP.py     # HRP 权重计算逻辑
│   └── example.py          # 如何继承 StrategyRunner 的示例
└── __init__.py
```

示例代码依赖 `Util_Fin` 模块族（`easy_manager`, `Position_util`, `logger_util` 等）与内部数据库 `macro_data_base`，请确认它们已安装并能在同一 Python 环境中导入。

## 环境准备

- Python ≥ 3.9
- 依赖（示例）：`pandas`、`numpy`、`scipy`、`matplotlib`、`seaborn`、`tqdm`、`openpyxl`
- 私有依赖：`Util_Fin`（提供数据、仓位、日志、评估工具）

安装公共依赖（若尚未安装）：

```bash
pip install pandas numpy scipy matplotlib seaborn tqdm openpyxl
```

## 快速开始

1. **配置数据与参数**打开 `example/HRP_config.py`，根据实际情况修改数据库、资产列表、回测窗口、日志/报表输出路径等。
2. **准备策略**`example/Strategy_HRP.py` 中的 `HRPSimpleBacktest.get_weight` 会基于历史收益的相关系数矩阵计算 HRP 权重，并返回 `N×1` 的矩阵。若要替换为其他策略，请确保实现同名方法，必要时实现 `get_strategy_info`/`get_other` 以在日志或自定义输出中使用。
3. **运行示例**

   ```bash
   # 根目录下
   python example/example.py
   ```

   或在代码中直接调用：

   ```python
   from NurlenSAA.main import StrategyRunner
   from NurlenSAA.example.HRP_config import HRPConfig, BenchmarkConfig

   runner = StrategyRunner(config=HRPConfig, benchmarks_config=BenchmarkConfig)
   runner.run_complete_workflow(plot=True)
   ```

   回测日志写入 `./log`，权重/收益/净值导出到 `./excel`，图表保存至 `./plot`。

## 案例配置说明

`HRP_config.HRPConfig` 控制全部回测参数，常用字段如下：

- `DATA_BASE` / `DATA_TABLE`：`Util_Fin.easy_manager` 将从该数据库表读取价格数据。
- `CODE_LIST`：资产代码序列，顺序需与策略输出权重一致。
- `BACKTEST_START_DATE` / `BACKTEST_END_DATE`：策略运行区间。
- `CHG_TIME_DELTA`、`INITIAL_MONTH`、`INITIAL_DAY`：调仓频率和首个调仓日。
- `CAL_WINDOW`：策略计算窗口（单位：交易日）。
- `SAVE_EVAL_RESULTS`、`EVAL_OUTPUT_PATH`、`PLOT_SAVE_PATH`：报表、图表的输出控制。
- `VAR_YEAR_WINDOWS`、`MAX_LOSS_LIMIT`：绩效评估时的风险参数。

`BenchmarkConfig` 用于声明基准组合，可一次启用多个固定权重方案，用于与策略结果同场比较。

## 扩展指南

1. **新增策略**创建继承自 `StrategyRunner` 的子类（参见 `example/example.py`），在其中重写：

   - `initialize_hrp_strategy`（或重命名为更通用的 `initialize_<your_strategy>`）以实例化自定义策略；
   - `initialize_strategies_agg` 将策略对象注册到 `self.strategies`；
   - 如需额外基准，调用 `self.benchmark_manager.add_benchmark`。
2. **编写策略逻辑**策略对象需提供：

   - `get_weight(ret_df, **kwargs)`：返回 `numpy.matrix` 类型的 `N×1` 权重；
   - （可选）`get_strategy_info()`：用于日志展示；
   - （可选）`get_other()`：用于保存额外结果（由 `BacktestEngine` 写入 `./excel/{strategy}_other_results.json`）。
3. **接入外部数据**
   若不用 `macro_data_base`，请重写 `StrategyRunner.load_data` 或修改 `easy_manager` 的配置，使其能够读取本地文件/接口。

## 输出与调试

- **日志**：`logger_util` 会在 `HRPConfig.LOG_FILE_PATH` 下生成以 `LOG_FILE_PREFIX` 命名的日志文件，记录每轮调仓权重及可能的异常。
- **Excel 报表**：`BacktestEngine` 自动导出 `weight_df`、`portfolio_returns`、`portfolio_pv`，`ReportGenerator` 还会输出整体绩效、年度分析等报表。
- **图表**：`StrategyRunner.plot_results` 生成净值曲线对比及权重堆叠图，默认保存在 `./plot`。

若运行失败，请先查看日志，再确认数据库连接与 `CODE_LIST` 数据完整性是否满足策略计算的窗口长度。

## 许可

尚未提供正式许可证，如需对外发布请先与项目所有者确认。
