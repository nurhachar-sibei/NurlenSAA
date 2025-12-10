from NurlenSAA.main import StrategyRunner
from HRP_config import HRPConfig,BenchmarkConfig
from Strategy_HRP import HRPSimpleBacktest

class Strategy_1(StrategyRunner):
    def __init__(self, config: HRPConfig, benchmarks: BenchmarkConfig):
        """
        初始化Strategy_1类
        :param config: HRP策略配置
        :param benchmarks: 基准策略配置
        """
        super().__init__(config, benchmarks)

    def initialize_hrp_strategy(self):
        """
        初始化HRP策略
        ---------------
        策略必须：
        有get_weights方法,输出Nx1的权重矩阵, 其中N为资产数量
        """
        self.hrp_strategy = HRPSimpleBacktest()
        strategy_info = self.hrp_strategy.get_strategy_info()
        self.logger.info(f"HRP策略初始化配置信息: {strategy_info}")

    def initialize_strategies_agg(self): #
        """
        初始化并收集所有的策略
        """
        self.initialize_hrp_strategy()
        self.strategies['HRP'] = self.hrp_strategy

        #其他策略
        ###############
if __name__ == "__main__":
    runner = Strategy_1(config=HRPConfig, benchmarks=BenchmarkConfig)
    
    # 运行完整工作流程
    success = runner.run_complete_workflow(
        plot=True  # 是否绘制图表
    )
    
    if success:
        print("\n程序执行成功!")
    else:
        print("\n程序执行失败,请查看日志了解详情。")

