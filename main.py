'''
模块集成并提供策略回测流程
'''

from Util_Fin.eval_module import ReportGenerator
import pandas as pd
import numpy as np
import warnings 
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

#导入案例配置用于__main__调试
from NurlenSAA.example.HRP_config import HRPConfig,BenchmarkConfig
from NurlenSAA.example.Strategy_HRP import HRPSimpleBacktest

#导入工具模块
from Util_Fin import easy_manager
from Util_Fin import Wind_util
from Util_Fin import Volatility_util
from Util_Fin import logger_util
from Util_Fin import Position_util
#导入策略模块
from NurlenSAA.benchmark_strategy import BenchmarkStrategy, BenchmarkManager
#导入回测引擎
from NurlenSAA.backtest_engine import BacktestEngine
#导入评价模块
from Util_Fin.eval_module import (
    PerformanceEvaluator,
    RiskAnalyzer,
    PeriodAnalyzer,
    ReportGenerator
)

class StrategyRunner:
    def __init__(self, config = None, benchmarks_config = None):
        '''
        初始化策略运行期
        config : 策略配置对象
        logger : 日志工具对象
        '''
        self.config = config 
        self.benchmarks_config = benchmarks_config
        self.logger = logger_util.setup_logger(
            log_file = self.config.LOG_FILE_PREFIX,
            file_load=self.config.LOG_FILE_PATH
        )

        #数据存储
        self.price_df = None
        self.ret_df=None
        self.position_df = None
        self.change_position_dates = None
        self.backtest_results = {}
        self.other_results = {}
        self.eval_results = {}

        self.strategies = {}

        self.logger.info("="*60)
        self.logger.info("策略运行器初始化完成")
        self.logger.info("="*60)

    def load_data(self):
        '''
        从数据库加载数据
        '''
        with easy_manager.EasyManager(database = self.config.DATA_BASE,
                                        logger_filename=self.config.LOG_FILE_PREFIX,
                                        logger_path=self.config.LOG_FILE_PATH,
                                        ) as em:
            self.price_df = em.load_table(self.config.DATA_TABLE,
                                        order_by='index',
                                        ascending=True,
                                        columns=self.config.CODE_LIST
                                        )
            self.price_df.set_index('index',inplace=True)
            self.price_df.index = pd.to_datetime(self.price_df.index)
            self.price_df = self.price_df.loc[self.config.DATA_START_DATE:]
            self.price_df.dropna(inplace=True)
        
        #计算收益率
        self.ret_df = self.price_df.pct_change()
        self.ret_df.dropna(inplace=True)
        self.logger.info(f"数据加载完成")
        self.logger.info(f"数据形状: {self.ret_df.shape}")
        self.logger.info(f"数据时间范围: {self.ret_df.index[0]} 到 {self.ret_df.index[-1]}")
        self.logger.info(f"资产列表: {self.config.CODE_LIST}")

    def prepare_position(self):
        """
        准备调仓信息
        ------------------
        get:
        ------------------
        self.position_df : pd.DataFrame
            以第一次开仓为起点的总持仓期内的基础池信息
            * 由于参数计算，注意开仓时间config.BACKTEST_START_DATE一般不是ret_df的第一个日期
        self.change_position_dates : list
            调仓日期列表
        self.change_position_df : pd.DataFrame
            仅获得换仓日时候的基础池信息(self.position_df的子集)
        """

        self.logger.info("准备持仓信息...")
        position_class = Position_util.Position_info(
            self.ret_df,
            self.config.BACKTEST_START_DATE,
            self.config.BACKTEST_END_DATE,
            self.config.CHG_TIME_DELTA,
            self.config.INITIAL_MONTH,
            self.config.INITIAL_DAY,
        )
        try:
            self.position_df,self.change_position_df,self.change_position_dates=\
                position_class.position_information()
            self.logger.info(f"持仓信息准备完成")
            self.logger.info(f"调仓日期范围: {self.change_position_dates[0]} 到 {self.change_position_dates[-1]}")
            self.logger.info(f"总调仓次数: {len(self.change_position_dates)}")
            #打印一下实例，方便判断调仓数据是否有问题
            if len(self.change_position_dates) >= 3:
                self.logger.info(f"调仓日期实例:")
                for i in range(min(3,len(self.change_position_dates))):
                    self.logger.info(f"{self.change_position_dates[i]}")
            self.change_position_df.to_excel(f"./excel/change_position_df.xlsx")
        except Exception as e:
            self.logger.error(f"持仓信息准备失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    #策略区
    def initialize_hrp_strategy(self):
        """
        初始化HRP策略
        """
        self.hrp_strategy = HRPSimpleBacktest()
        strategy_info = self.hrp_strategy.get_strategy_info()
        self.logger.info(f"HRP策略初始化配置信息: {strategy_info}")

    def initialize_benchmarks(self):
        """
        初始化基准策略
        """
        self.logger.info("初始化benchmark策略...")
        self.benchmark_manager = BenchmarkManager()

        #从配置中创建benchmarks
        enabled_benchmarks = self.benchmarks_config.get_enabled_benchmarks()
        
        for name, config in enabled_benchmarks.items():
            benchmark = BenchmarkStrategy(
                name=name,
                weights=config['weights'],
                description=config['name']
            )
            self.benchmark_manager.add_benchmark(benchmark)
            self.logger.info(f"添加Benchmark: {name} - {config['name']}")
    
    def initialize_strategies_agg(self):
        """
        初始化并收集所有的策略
        """
        self.initialize_hrp_strategy()
        self.strategies['HRP'] = self.hrp_strategy

        #其他策略
        ###############



    def run_backtest(self):
        """
        运行HRP策略回测
        """
        self.logger.info("="*60)
        self.logger.info("开始运行回测")
        self.logger.info("="*60)

        #创建回测引擎
        engine = BacktestEngine(logger=self.logger)

        #策略字典
        strategies = self.strategies
        strategy_params = {}
        #非标准参数
        #strategy_params['HRP'] = {'xxx':xxxx}
        #benchmark字典
        if self.benchmark_manager:
            for name,benchmark in self.benchmark_manager.get_all_benchmarks().items():
                strategies[name] = benchmark
                # Benchmark使用简单的协方差计算器(实际上不需要)


        #
        #运行回测
        results,other_results = engine.run_multi_strategy_backtest(
            strategies_dict=strategies,
            position_df=self.position_df,
            change_position_dates=self.change_position_dates,
            ret_df=self.ret_df,
            cal_windows=self.config.CAL_WINDOW,
            strategy_params_dict=strategy_params
        )

        self.backtest_results = results
        self.other_results = other_results

        self.logger.info( "="*60)
        self.logger.info("回测完成")
        self.logger.info("="*60)

                # 打印简要结果
        summary = engine.get_results_summary()
        if summary is not None:
            self.logger.info("\n回测结果摘要:")
            self.logger.info("\n" + summary.to_string())

    def evaluate_results(self, save_results=True):
        """评价回测结果"""
        self.logger.info( "="*60)
        self.logger.info("开始评价回测结果")
        self.logger.info("="*60)
        
        # 提取净值序列
        pv_dict = {}
        for name, result in self.backtest_results.items():
            pv_dict[name] = result['portfolio_pv']
        
        # 整体评价
        eval_df = PerformanceEvaluator.evaluate_multi_portfolios(pv_dict)
        self.eval_results['overall'] = eval_df
        
        # 打印整体评价
        ReportGenerator.print_performance_report(eval_df, '策略绩效评价')
        
        # 策略的年度分析
        for name, result in self.backtest_results.items():
            rp_returns = result['portfolio_returns']
            annual_df = PeriodAnalyzer.annual_analysis(
                rp_returns,
                var_windows=self.config.VAR_YEAR_WINDOWS,
                max_loss_limit=self.config.MAX_LOSS_LIMIT
            )
            self.eval_results['annual'] = annual_df
            
            # 打印年度报告
            ReportGenerator.print_annual_report(annual_df, f'{name}策略年度分析')
        
        # 保存结果
        if save_results and self.config.SAVE_EVAL_RESULTS:
            import os
            if not os.path.exists(self.config.EVAL_OUTPUT_PATH):
                os.makedirs(self.config.EVAL_OUTPUT_PATH)
            
            ReportGenerator.save_report(
                eval_df,
                self.eval_results.get('annual'),
                self.config.EVAL_OUTPUT_PATH,
                filename_prefix=self.config.EVAL_FILENAME_PREFIX
            )
    
    def plot_results(self):
        """绘制结果图表"""
        self.logger.info("\n绘制结果图表...")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 净值曲线对比
        import os
        plot_dir = "./plot"
        os.makedirs(plot_dir, exist_ok=True)

        # 1. 绘制并保存净值曲线对比图
        fig, ax = plt.subplots(figsize=(15, 8))
        for name, result in self.backtest_results.items():
            pv = result['portfolio_pv']
            ax.plot(pv.index, pv.values, label=name, linewidth=2)

        ax.set_title('策略净值曲线对比', fontsize=14, fontweight='bold')
        ax.set_xlabel('日期')
        ax.set_ylabel('净值')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "净值曲线对比.png"), dpi=300)
        plt.show()

        # 2. 逐个保存各策略权重变化图
        for name, result in self.backtest_results.items():
            if 'weight_df' not in result:
                continue
            weight_df = result['weight_df']
            fig, ax = plt.subplots(figsize=(15, 6))
            ax.stackplot(
                weight_df.index,
                weight_df.T,
                labels=weight_df.columns,
                alpha=0.8
            )
            ax.set_title(f'{name}策略权重变化', fontsize=14, fontweight='bold')
            ax.set_xlabel('日期')
            ax.set_ylabel('权重')
            ax.set_yticks(np.arange(0, 1.1, 0.2))
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{name}_权重变化.png"), dpi=300)
            plt.close()

    def run_complete_workflow(self, plot=True):
        """
        运行完整的策略工作流程
        
        Parameters:
        -----------
        plot : bool
            是否绘制图表
        """
        self.logger.info("\n")
        self.logger.info("="*80)
        self.logger.info("回测系统-开始".center(80))
        self.logger.info("="*80 + "\n")


        try:
            # 1. 加载数据
            self.load_data()

            
            # 2. 准备调仓信息
            self.prepare_position()
            
            # 3. 初始化策略
            self.initialize_benchmarks()
            self.initialize_strategies_agg()
            
            # 4. 运行回测
            self.run_backtest()
            
            # 5. 评价结果
            self.evaluate_results(save_results=self.config.SAVE_EVAL_RESULTS)
            
            # 6. 绘制图表
            if plot:
                self.plot_results()
            
            self.logger.info("="*60)
            self.logger.info("完整工作流程执行完成!")
            self.logger.info("="*60)
            
            return True
            
        except Exception as e:
            self.logger.error(f"工作流程执行失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False





if __name__ == "__main__":
    runner = StrategyRunner(config=HRPConfig, benchmarks_config=BenchmarkConfig)
    
    # 运行完整工作流程
    success = runner.run_complete_workflow(
        plot=True  # 是否绘制图表
    )
    
    if success:
        print("\n程序执行成功!")
    else:
        print("\n程序执行失败,请查看日志了解详情。")




