#!/usr/bin/env python
# coding: utf-8
"""
Benchmark策略模块
提供固定权重的基准策略
"""

import numpy as np
import pandas as pd


class BenchmarkStrategy:
    """基准策略类"""
    
    def __init__(self, name, weights, description=''):
        """
        初始化基准策略
        
        Parameters:
        -----------
        name : str
            基准策略名称
        weights : list or np.array
            固定权重向量
        description : str
            策略描述
        """
        self.name = name
        self.weights = np.array(weights) if not isinstance(weights, np.ndarray) else weights
        self.description = description
        
        # 验证权重
        self._validate_weights()
    
    def _validate_weights(self):
        """验证权重是否合法"""
        # 检查权重和是否为1
        weight_sum = np.sum(self.weights)
        if not np.isclose(weight_sum, 1.0, rtol=1e-5):
            raise ValueError(f"权重和必须为1,当前为{weight_sum}")
        
        # 检查权重是否非负
        if np.any(self.weights < 0):
            raise ValueError("权重不能为负数")
    
    def get_weight(self, ret_df=None, *args, **kwargs):
        """
        获取基准策略权重
        
        Parameters:
        -----------
        ret_df : pd.DataFrame, optional
            收益率数据(对于固定权重策略不需要,但为了接口统一保留此参数)
        
        Returns:
        --------
        weight : np.matrix
            权重矩阵 (n×1)
        """
        return np.matrix(self.weights).T
    
    def get_strategy_info(self):
        """获取策略信息"""
        return {
            'name': self.name,
            'description': self.description,
            'weights': self.weights.tolist(),
            'type': 'Fixed Weight Benchmark'
        }
    
    def __repr__(self):
        return f"BenchmarkStrategy(name='{self.name}', weights={self.weights.tolist()})"


class DynamicBenchmarkStrategy(BenchmarkStrategy):
    """动态基准策略类(可扩展用于实现动态调整的基准)"""
    
    def __init__(self, name, weights, rebalance_method='none', description=''):
        """
        初始化动态基准策略
        
        Parameters:
        -----------
        name : str
            基准策略名称
        weights : list or np.array
            初始权重向量
        rebalance_method : str
            再平衡方法
            'none': 不再平衡(买入持有)
            'periodic': 定期再平衡到初始权重
            'threshold': 偏离阈值时再平衡
        description : str
            策略描述
        """
        super().__init__(name, weights, description)
        self.rebalance_method = rebalance_method
        self.initial_weights = self.weights.copy()
    
    def get_weight(self, ret_df=None, current_weights=None, threshold=0.05):
        """
        获取动态基准策略权重
        
        Parameters:
        -----------
        ret_df : pd.DataFrame, optional
            收益率数据
        current_weights : np.array, optional
            当前权重
        threshold : float
            再平衡阈值(用于threshold方法)
        
        Returns:
        --------
        weight : np.matrix
            权重矩阵 (n×1)
        """
        if self.rebalance_method == 'none':
            # 买入持有,不调整
            if current_weights is not None:
                return np.matrix(current_weights).T
            else:
                return np.matrix(self.initial_weights).T
        
        elif self.rebalance_method == 'periodic':
            # 定期再平衡到初始权重
            return np.matrix(self.initial_weights).T
        
        elif self.rebalance_method == 'threshold':
            # 偏离阈值时再平衡
            if current_weights is None:
                return np.matrix(self.initial_weights).T
            
            # 计算偏离度
            deviation = np.abs(current_weights - self.initial_weights)
            max_deviation = np.max(deviation)
            
            if max_deviation > threshold:
                # 超过阈值,再平衡
                return np.matrix(self.initial_weights).T
            else:
                # 未超过阈值,保持当前权重
                return np.matrix(current_weights).T
        
        else:
            # 默认返回初始权重
            return np.matrix(self.initial_weights).T
    
    def get_strategy_info(self):
        """获取策略信息"""
        info = super().get_strategy_info()
        info.update({
            'type': 'Dynamic Benchmark',
            'rebalance_method': self.rebalance_method
        })
        return info


class BenchmarkManager:
    """基准策略管理器"""
    
    def __init__(self):
        """初始化基准策略管理器"""
        self.benchmarks = {}
    
    def add_benchmark(self, benchmark):
        """
        添加基准策略
        
        Parameters:
        -----------
        benchmark : BenchmarkStrategy
            基准策略对象
        """
        if not isinstance(benchmark, BenchmarkStrategy):
            raise TypeError("必须是BenchmarkStrategy类型")
        
        self.benchmarks[benchmark.name] = benchmark
    
    def remove_benchmark(self, name):
        """
        移除基准策略
        
        Parameters:
        -----------
        name : str
            基准策略名称
        """
        if name in self.benchmarks:
            del self.benchmarks[name]
        else:
            print(f"警告: 基准策略'{name}'不存在")
    
    def get_benchmark(self, name):
        """
        获取基准策略
        
        Parameters:
        -----------
        name : str
            基准策略名称
        
        Returns:
        --------
        benchmark : BenchmarkStrategy
            基准策略对象
        """
        if name in self.benchmarks:
            return self.benchmarks[name]
        else:
            raise KeyError(f"基准策略'{name}'不存在")
    
    def list_benchmarks(self):
        """列出所有基准策略"""
        print("\n当前配置的基准策略:")
        print("=" * 60)
        for name, benchmark in self.benchmarks.items():
            info = benchmark.get_strategy_info()
            print(f"名称: {name}")
            print(f"描述: {info['description']}")
            print(f"权重: {info['weights']}")
            print(f"类型: {info['type']}")
            print("-" * 60)
    
    def get_all_benchmarks(self):
        """获取所有基准策略的字典"""
        return self.benchmarks
    
    def create_from_config(self, benchmark_configs):
        """
        从配置字典创建基准策略
        
        Parameters:
        -----------
        benchmark_configs : dict
            基准配置字典,格式:
            {
                'name1': {'name': '显示名称', 'weights': [...], 'enabled': True},
                'name2': {...}
            }
        """
        for key, config in benchmark_configs.items():
            if config.get('enabled', False):
                benchmark = BenchmarkStrategy(
                    name=key,
                    weights=config['weights'],
                    description=config.get('name', '')
                )
                self.add_benchmark(benchmark)


if __name__ == '__main__':
    # 测试代码
    print("Benchmark策略模块加载成功!")
    
    # 创建基准策略管理器
    manager = BenchmarkManager()
    
    # 添加几个基准策略
    bench_8020 = BenchmarkStrategy(
        name='8020',
        weights=[0.8, 0.1, 0.1],
        description='80%股票 20%其他'
    )
    
    bench_9010 = BenchmarkStrategy(
        name='9010',
        weights=[0.9, 0.05, 0.05],
        description='90%股票 10%其他'
    )
    
    manager.add_benchmark(bench_8020)
    manager.add_benchmark(bench_9010)
    
    # 列出所有基准
    manager.list_benchmarks()
    
    # 获取权重
    print("\n获取8020基准的权重:")
    weight = bench_8020.get_weight()
    print(weight.T)
    
    # 测试动态基准
    print("\n测试动态基准策略:")
    dynamic_bench = DynamicBenchmarkStrategy(
        name='dynamic_8020',
        weights=[0.8, 0.1, 0.1],
        rebalance_method='threshold',
        description='动态再平衡基准'
    )
    
    # 模拟当前权重偏离
    current_weights = np.array([0.85, 0.08, 0.07])
    print(f"当前权重: {current_weights}")
    print(f"初始权重: {dynamic_bench.initial_weights}")
    
    # 获取调整后的权重(阈值0.05)
    adjusted_weight = dynamic_bench.get_weight(
        current_weights=current_weights, 
        threshold=0.05
    )
    print(f"调整后权重: {adjusted_weight.T}")

