'''
配置文件 - 管理HRP所使用的参数
'''

class HRPConfig:
    # START_DATE = '2005-01-01'
    # END_DATE = '2024-03-15'
    # CHG_FREQ = 'MS'

    #===========日志文件配置===============
    #日志文件路径
    LOG_FILE_PATH = './log'
    #日志文件名前缀
    LOG_FILE_PREFIX = 'HRP'
    #日志级别
    LOG_LEVEL = 'INFO'
    #===========数据读取相关配置============
    DATA_BASE = 'macro_data_base'
    DATA_TABLE = 'daily_asset_price_1' #价格数据

    CODE_LIST = [
        '000300_SH',
        '000905_SH',
        'CBA00601_CS',
        'CBA02001_CS',
        '000832_CSI',
        'NH0200_NHF',
        'NH0300_NHF',
        'B_IPE',
        'AU9999_SGE',
        # 'USDCH_FX'
    ]
    #数据库读取开始时间
    DATA_START_DATE = '2006-01-01'
    #===========回测相关配置===============
    #回测开始时间
    BACKTEST_START_DATE = '2007-07-01'
    #回测结束时间
    BACKTEST_END_DATE = '2025-11-01'
    #调仓频率
    CHG_TIME_DELTA = 'MS'
    # 初始调仓月份(年度调仓时使用)
    INITIAL_MONTH = 1
    # 初始调仓日(年度调仓时使用)
    INITIAL_DAY = 1
    # 计算窗口
    CAL_WINDOW = 126
    # ============ 输出配置 ============
    # 评价结果输出目录
    EVAL_OUTPUT_PATH = './excel'
    EVAL_FILENAME_PREFIX = './HRP_strategy_report'
    
    # 是否保存评价结果
    SAVE_EVAL_RESULTS = True
    
    # 图表保存路径
    PLOT_SAVE_PATH = './plots'
    # ============ 评价指标参数 ============
    # VaR计算的历史窗口(年数)
    VAR_YEAR_WINDOWS = 5
    
    # VaR置信水平
    VAR_CONFIDENCE_LEVEL = 0.95
    
    # 最大损失限额(用于计算最大投资规模, 单位:元)
    MAX_LOSS_LIMIT = 200_000_000


class BenchmarkConfig:
    '''
    benchmark策略配置类
    '''
    BENCHMARKS = {
        'FixRatio':{
            'name':'FixRatio',
            'weights':[0.1, 0.1,0.25,0.2,0.1,0.05,0.05,0.05,0.1],
            'enabled':True
        }
    }

    @classmethod
    def get_enabled_benchmarks(cls):
        """获取启用的benchmark配置"""
        return {
            name: config for name, config in cls.BENCHMARKS.items() 
            if config['enabled']
        }



