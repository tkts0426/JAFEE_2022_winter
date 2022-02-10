import numpy as np
import pandas as pd
import math

np.random.seed(1234)

class CalculateSpotVolatility(): 
    def __init__(self, n, T, alpha, beta, rho, X_0, vol_0, e, K, N):
        # parameter for simulating SABR model
        self.n_points = n
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.X_0 = X_0
        self.vol_0 = vol_0
        
        # parameter for calculating spot volatility
        self.y = np.random.normal(loc = 0, scale = e , size = 10000)
        self.K = K
        self.N = N
        
    def _simulate_BM(self):
        '''
            Simulate a Brownian motion
        '''
        self.delta_t = 1. * self.T/self.n  # decide the increment of the time
        partition = [i * self.delta_t for i in range(self.n + 1)] # make a partition
        
        # difference of BM（loc:mean，scale:standard deviation, size:the number of points）
        random_increments = np.random.normal(loc = 0.0, scale = np.sqrt(delta_t), size = self.n)
        # culculate Brownian motion
        brownian_motion = np.r_[0., np.cumsum(random_increments)]

        return brownian_motion, random_increments, partition
    
    def _calculate_SABRvol(self):
        self.vols_root = [self.vol_0]
        for i in range(1, self.n + 1):
            pre_vol = self.vols_root[i-1]
            vol_root = pre_vol * math.exp((1/2) * self.alpha * (self.rho * self.dB_1[i-1] + np.sqrt(1 - self.rho**2) * self.dB_2[i-1]) - (1/4) * (self.alpha**2) * self.delta_t)
            self.vols_root.append(vol_root)
    
    def _simulate_SABR(self):
        '''
            Simulate a SABR model by Eular-Maruyama method
        '''
        BM_1, self.dB_1, partition = self._simulate_BM()
        BM_2, self.dB_2, partition = self._simulate_BM()
        
        # SABR model に従う ”Process X”と ”volatility sig” を作成
        X = np.zeros(self.n + 1)
        X[0] = self.X_0
        
        # calculate the volatility process directly
        self._calculate_SABRvol()
        
        for i, dB_1_t, dB_2_t, t in zip(range(1, self.n+1), self.dB_1, self.dB_2, partition):
            # 1つ前の X と sig の値
            pre_vol = self.vols_root[i-1]
            pre_X = X[i-1] 
            # X と sig の値を計算（SDEに従う）
            X[i] = pre_X + pre_vol * pre_X**(self.beta) * dB_1_t
        
        data_array = np.c_[partition, X, self.vols_root]
        self.df = pd.DataFrame(data_array, columns = ['timestamp', 'process', 'volatility'])
        
        # return self.df

    def _calculate_Dirichlet_kernel(self, x):
        return 1/(2*self.N + 1) * (np.sin(2 * np.pi * (self.N + 1/2) * x)) / (np.sin(2 * np.pi * x / 2) ) 
    
    def calculate_spot_volatility(self):
        partition = [i/self.n for i in range(self.n)]
        self._simulate_SABR()
        X = self.df['process'].values
    
        spot_vol_process = []
        #各時刻tに対して分割i,jに対するスポットボラティリティを計算する
        #nは分割数
        for t in partition:
            total_i = 0
            for i in range(self.K):
                add_j = 0
                for j in range(self.n):
                    if (np.sin(2 * np.pi* (t - partition[j] + self.y[i]) /2)  == 0):
                        D_ = 1
                    else:
                        D_ = self._calculate_Dirichlet_kernel(t - partition[j] + self.y[i])
                    
                    add_j += D_ * (X[j+1] - X[j])
                    
                total_i += add_j**2

            spot_vol = (2*self.N + 1) * total_i / self.K
            
            spot_vol_process.append(spot_vol)

        #個数を合わせる為
        self.df = self.df[:-1]

        self.df['spot volatility'] = spot_vol_process
        
        # if beta != 0:
        #     df_path['abs_error_vol'] = np.abs((df_path['process'] * df_path['volatility'])**2 - df_path['spot volatility'])
        # else:
        #     df_path['abs_error_vol'] = np.abs(df_path['volatility']**2 - df_path['spot volatility'])

        return self.df
