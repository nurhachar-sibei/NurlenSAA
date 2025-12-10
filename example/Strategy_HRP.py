import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import seaborn as sns

from Util_Fin import Wind_util
from Util_Fin import easy_manager
from NurlenSAA.example.HRP_config import HRPConfig


class HRPSimpleBacktest:
    def __init__(self, price_data=None):
        self.name = 'HRP_strategy'
    def get_strategy_info(self):
        """
        获取策略信息
        """
        return {
            "strategy_name": self.name,
        }
    
    def distance_cal(self):
        '''
        距离矩阵计算
        '''
        self.D_g = np.sqrt(0.5 * (1 - self.rho_g))
        for i in range(len(self.D_g)):
            self.D_g.iloc[i,i] = 0
    
    def linkage_get(self):
        '''
        聚类链接矩阵计算
        '''
        # print(f'距离矩阵:\n{self.D_g}')
        self.Z = linkage(self.D_g, method = 'ward',metric = 'euclidean')
        # dn=dendrogram(self.Z)
        # plt.show()
    
    def get_qusi_diag(self):
        '''
        拟对角化
        '''
        link = self.Z
        link = link.astype(int)
    
        # get the first and the second item of the last tuple
        sort_ix = pd.Series([link[-1,0], link[-1,1]]) 
        
        # the total num of items is the third item of the last list
        num_items = link[-1, 3]
        
        # if the max of sort_ix is bigger than or equal to the max_items
        while sort_ix.max() >= num_items:
            # assign sort_ix index with 24 x 24
            sort_ix.index = range(0, sort_ix.shape[0]*2, 2) # odd numers as index
            
            df0 = sort_ix[sort_ix >= num_items] # find clusters
            
            # df0 contain even index and cluster index
            i = df0.index
            j = df0.values - num_items # 
            
            sort_ix[i] = link[j,0] # item 1
            
            df0  = pd.Series(link[j, 1], index=i+1)
            
            # sort_ix = sort_ix.append(df0)
            sort_ix = pd.concat([sort_ix, df0])

            sort_ix = sort_ix.sort_index()
            
            sort_ix.index = range(sort_ix.shape[0])
        self.sort_ix = sort_ix.tolist()
    def _get_cluster_var(self, cov, c_items):
        cov_ = cov.iloc[c_items, c_items] # matrix slice
        # calculate the inversev-variance portfolio
        ivp = 1./np.diag(cov_)
        ivp/=ivp.sum()
        w_ = ivp.reshape(-1,1)
        c_var = np.dot(np.dot(w_.T, cov_), w_)[0,0]
        return c_var
    def hrp_weights(self, cov, sort_ix):
        w = pd.Series(1, index=sort_ix)
        c_items = [sort_ix] # initialize all items in one cluster
        while len(c_items) > 0:
            c_items = [i[j:k] for i in c_items for j,k in ((0,len(i)//2), (len(i)//2, len(i))) if len(i) > 1]
            for i in range(0, len(c_items), 2):
                c_items0 = c_items[i]
                c_items1 = c_items[i+1]
                c_var0 = self._get_cluster_var(cov, c_items0)
                c_var1 = self._get_cluster_var(cov, c_items1)
                alpha = 1 - c_var0/(c_var0 + c_var1)
                w[c_items0] *= alpha
                w[c_items1] *= 1 - alpha
        return w
    def cov_cal(self):
        self.cov = self.ret_data.cov()



    def get_weight(self,ret_data):
        self.ret_data = ret_data
        self.ret_data_mark = self.ret_data.copy()
        self.ret_data_mark.columns = range(len(self.ret_data.columns))
        self.rho_g = self.ret_data.corr()
        self.distance_cal()
        self.linkage_get()
        self.get_qusi_diag()
        self.cov_cal()
        self.weights = self.hrp_weights(self.cov, self.sort_ix)

        new_index = [self.ret_data.columns[i] for i in self.sort_ix]
        self.weights.index = new_index
        self.weights = self.weights[HRPConfig.CODE_LIST]
        self.weights_matrix = np.matrix(self.weights).T #输出对应columns资产的Nx1权重矩阵
        return self.weights_matrix
    # def get_other(self):
    #     return {'weight':self.weights_matrix}
if __name__ == '__main__':
    with easy_manager.EasyManager(database = 'macro_data_base') as em:
        price_data = em.load_table("daily_asset_price_1")
        price_data.set_index('index',inplace=True)
        price_data.index = pd.to_datetime(price_data.index)
        price_data = price_data[HRPConfig.CODE_LIST]
        ret_data = price_data.pct_change().dropna()
    hrp_bt = HRPSimpleBacktest()
    w = hrp_bt.get_weight(ret_data)
    print(hrp_bt.weights)
    print(hrp_bt.weights_matrix)
  





