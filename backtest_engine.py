'''
回测引擎模块
提供通用的回测框架，用于运行和评估策略。
'''

import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import json
import datetime
import warnings
# 忽略 FutureWarning 类型的警告
warnings.simplefilter(action='ignore', category=FutureWarning)

class BacktestEngine:
    '''
    回测引擎类
    '''

    def __init__(self, logger=None):
        '''
        初始化回测引擎
        Parameters:
        -----------
        logger: logging.Logger, optional
            日志记录器，用于记录回测过程中的信息。如果未提供，将创建一个默认的日志记录器。
        '''
        self.logger = logger or logging.getLogger(__name__)
        self.other_result = {}

    def calculate_dynamic_weight(self,hold_df,init_weight,init_date):
        '''
        计算因为价格变动而非调仓变动导致的权重变动
        Parameters:
        -----------
        hold_df: pd.DataFrame
            持仓收益率数据框，包含每个资产的持仓收益率。
        init_weight: pd.Series
            调仓日第一天所持有的权重(nx1矩阵)
        init_date: str or datetime.date
            调仓第一日对应的日期。
        Returns:
        --------
        weight_df: pd.DataFrame
            动态权重数据框，包含本次调仓持有期内每个资产的动态权重。
        ''' 
        weights_df = pd.DataFrame(
            index = hold_df.index,
            columns = hold_df.columns,
        ).fillna(0)

        #设置初始权重
        weights_df.loc[init_date] = init_weight.T.tolist()[0]
        prev_weights = weights_df.loc[init_date].values

        #逐日更新权重
        for i in range(1,len(weights_df)):
            daily_ret = hold_df.iloc[i].values
            # 计算组合总收益率
            port_return = np.dot(prev_weights, daily_ret)
            # 计算新权重
            new_weights = prev_weights * (1 + daily_ret) / (1 + port_return)
            weights_df.iloc[i] = new_weights
            prev_weights = new_weights

        return weights_df

    def run_backtest(self,
                     strategy,
                     position_df,
                     change_position_dates,
                     ret_df,
                     cal_windows,
                     **strategy_params):
        '''
        运行回测
        Parameters:
        -----------
        strategy: Strategy
            策略对象，输出调仓期权重。
        position_df: pd.DataFrame
            持仓期数据框，包含每个资产的持仓数量。
        change_position_dates: list of str or datetime.date
            调仓日期列表，包含了每个调仓日的日期。
        ret_df: pd.DataFrame
            收益率数据框，包含每个资产的收益率。
        cal_windows: int
            计算窗口大小，即每次调仓时考虑的历史数据窗口大小。
        **strategy_params: dict
            策略参数，用于传递给策略对象的参数。
        Returns:
        --------
        results: dict
            回测结果字典，包含了回测的各种指标和结果。
        '''
        total_weights_df = None

        #事件驱动核心循环
        for i in range(len(change_position_dates)):
            #获取当前调仓日
            p_d = change_position_dates[i]
            #确定下一个调仓日
            try:
                next_p_d = change_position_dates[i+1]
                #获取策略计算期的输入数据 -> 必须为过去的数据
                cal_df = ret_df.loc[:p_d].iloc[-cal_windows-1:-1]
                #获取当前调仓日的持仓数据
                hold_df = position_df.loc[p_d:next_p_d].iloc[:-1]
            except:
                next_p_d = ret_df.index[-1]
                #获取策略计算期的输入数据 -> 必须为过去的数据
                cal_df = ret_df.loc[:p_d].iloc[-cal_windows-1:-1]
                #获取当前调仓日的持仓数据
                hold_df = position_df.loc[p_d:next_p_d]

            ##策略权重计算
            #策略核心识别
            if hasattr(strategy,'get_weight'):
                #调用策略权重计算方法
                weight = strategy.get_weight(cal_df,**strategy_params) #策略输出weight必须为N*1的矩阵
            else:
                raise AttributeError('策略对象必须实现get_weight方法')

            #一些可能需要的参数输出
            if hasattr(strategy,'get_other'):
                other = strategy.get_other()
                self.other_result[str(p_d)] = other
            else:
                self.other_result[str(p_d)] = {}
            
            self.logger.info(
                    f"回测日期: {p_d.strftime('%Y-%m-%d')}, "
                    f"权重: {weight.T.tolist()[0]}"
            )

            #计算动态权重
            dynamic_weights_df = self.calculate_dynamic_weight(hold_df,weight,p_d)
            #每日权重表格
            if total_weights_df is None:
                total_weights_df = dynamic_weights_df
            else:
                total_weights_df = pd.concat([total_weights_df,dynamic_weights_df])
        #计算组合收益率
        asset_returns = position_df * total_weights_df
        portfolio_returns = asset_returns.sum(axis=1)
        portfolio_pv = (1+portfolio_returns).cumprod()

        #结果
        result = {
            'weight_df':total_weights_df,
            'asset_returns':asset_returns,
            'portfolio_returns':portfolio_returns,
            'portfolio_pv':portfolio_pv,
        }

        return result,self.other_result
    def run_multi_strategy_backtest(self,
                                    strategies_dict,
                                    position_df,
                                    change_position_dates,
                                    ret_df,
                                    cal_windows,
                                    strategy_params_dict=None):
        '''
        运行多策略回测
        Parameters:
        -----------
        strategies_dict: dict
            策略字典，包含了多个策略对象。
        position_df: pd.DataFrame
            持仓期数据框，包含每个资产的持仓数量。
        change_position_dates: list of str or datetime.date
            调仓日期列表，包含了每个调仓日的日期。
        ret_df: pd.DataFrame
            收益率数据框，包含每个资产的收益率。
        cal_windows: int
            计算窗口大小，即每次调仓时考虑的历史数据窗口大小。
        strategy_params_dict: dict of dict, optional
            策略参数字典，包含了每个策略的参数。
        Returns:
        --------
        results: dict
            回测结果字典，包含了每个策略的回测结果。
        '''
        results = {}
        other_results = {} #一些非标准变量的输出
        for strategy_name,strategy in strategies_dict.items():
            self.logger.info(f"{'='*60}")
            self.logger.info(f"开始回测策略: {strategy_name}")
            self.logger.info(f"{'='*60}")

            #获取该策略的参数
            strategy_params = strategy_params_dict.get(strategy_name, {}) if strategy_params_dict else {}

            #运行回测
            try:
                result,other_result = self.run_backtest(
                    strategy=strategy,
                    position_df=position_df,
                    change_position_dates=change_position_dates,
                    ret_df=ret_df,
                    cal_windows=cal_windows,
                    **strategy_params
                )
                results[strategy_name] = result
                other_results[strategy_name] = other_result
                # current_date = datetime.datetime.now()
                result['weight_df'].to_excel(f'./excel/{strategy_name}_weight_df.xlsx')
                result['portfolio_returns'].to_excel(f'./excel/{strategy_name}_portfolio_returns.xlsx')
                result['portfolio_pv'].to_excel(f'./excel/{strategy_name}_portfolio_pv.xlsx')
                if hasattr(strategy,'get_other'):
                    with open(f"./excel/{strategy_name}_other_results.json", "w") as f:
                        json.dump(other_result, f, indent=4)
                self.logger.info(f"策略'{strategy_name}'回测完成")


            except Exception as e:
                self.logger.error(f"策略 {strategy_name} 回测运行出错: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
        self.results =results
        return results,other_results
    def get_results_summary(self):
        """
        获取回测结果摘要
        
        Returns:
        --------
        summary : pd.DataFrame
            结果摘要表格
        """
        if not self.results:
            print("尚未运行回测或回测结果为空")
            return None
        
        summary_data = []
        for strategy_name, result in self.results.items():
            pv = result['portfolio_pv']
            returns = result['portfolio_returns']
            
            summary_data.append({
                '策略名称': strategy_name,
                '最终净值': pv.iloc[-1],
                '累计收益率': pv.iloc[-1] / pv.iloc[0] - 1,
                '年化收益率': (pv.iloc[-1] / pv.iloc[0]) ** (252 / len(pv)) - 1,
                '年化波动率': returns.std() * np.sqrt(252),
                '最大回撤': self._calculate_max_drawdown(returns)
            })
        
        return pd.DataFrame(summary_data)
    
    def _calculate_max_drawdown(self, returns):
        """
        计算最大回撤
        
        Parameters:
        -----------
        returns : pd.Series
            收益率序列
            
        Returns:
        --------
        max_dd : float
            最大回撤
        """
        cumulative_wealth = (1 + returns).cumprod()
        running_max = cumulative_wealth.expanding().max()
        drawdown = (cumulative_wealth - running_max) / running_max
        max_drawdown = drawdown.min()
        return max_drawdown


if __name__ == '__main__':
    # 测试代码
    print("回测引擎模块加载成功!")
    
    # 创建示例数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=20, freq='D')
    assets = ['股票', '债券', '黄金']
    returns = pd.DataFrame(
        np.random.randn(20, 3) * 0.01,
        index=dates,
        columns=assets
    )
    
    # 创建简单的测试策略
    class SimpleStrategy:
        def get_weight(self, ret_df):
            return np.matrix([0.6, 0.3, 0.1]).T
    
    # 创建回测引擎
    engine = BacktestEngine()
    
    print("\n回测引擎创建成功!")
    print(f"示例数据形状: {returns.shape}")
    print(f"数据时间范围: {returns.index[0]} 到 {returns.index[-1]}")
    print("\n运行回测...")
    results,other_results = engine.run_multi_strategy_backtest(
        strategies_dict={
            'SimpleStrategy': SimpleStrategy(),
        },
        position_df=returns,
        change_position_dates=returns.index,
        ret_df=returns,
        cal_windows=1
    )
    print("\n回测结果摘要:")
    print(engine.get_results_summary())



