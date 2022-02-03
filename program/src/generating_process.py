import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# 乱数の発生の際に必要な設定（固定しておかないと，実装ごとにずれが出る）
np.random.seed(1234)

# ブラウン運動を発生させる関数
'''
入力：
    N：分割の個数
    T：時刻の区間
出力：
    brownian_motion：ブラウン運動のPath
    random_increments：ブラウン運動の差分（N(0,T/N)の正規分布に従うように発生）
    partition：時刻の分割
'''
def brownian_motion(n, T):
    '''
    Simulates a Brownian motion
    :param int N : the number of discrete steps
    :param int T: the number of continuous time steps
    :param float h: the variance of the increments
    '''
    delta_t = 1. * T/n  # decide the increment of the time
    partition = [i*delta_t for i in range(n + 1)] # make a partition
    
    # ブラウン運動の差分（平均：０，標準偏差：時間の差分）
    random_increments = np.random.normal(loc = 0.0, scale = np.sqrt(delta_t), size = n)
    '''
    where loc = "mean(average)", scale = "variance", N = the number of increments.
    (正規分布を発生させている)
    '''
    # making data like a Brownian motion
    brownian_motion = np.cumsum(random_increments)  # calculate the brownian motion
    # insert the initial condition
    brownian_motion = np.insert(brownian_motion, 0, 0.0)
    
    return brownian_motion, random_increments, partition


# SABR model に従うProcessの生成
'''
入力：
    N，T：分割幅と時間（区間[0,T]のこと）
    alpha：volatility of volatility proccess(in general say vol-vol)
    beta：process が変化するときにアットザマネーのボラティリティーがどのように変化するか
    rho：２つのブラウン運動の相関係数
    X_0, sig_0：それぞれの process の初期値
出力：
    df <=> [columns] partition, process, volatility process
'''
def simulate_SABR(n = 100, T = 1, alpha = 0.3, beta = 1, rho = 0.5, X_0 = 0.1, vol_0 = 0.1, fig_mode = False):
    # BMの生成
    BM_1, dB_1, partition = brownian_motion(n, T)
    BM_2, dB_2, partition = brownian_motion(n, T)
    dt = 1. * T / n
    
    # SABR model に従う ”Process X”と ”volatility sig” を作成
    X = np.zeros(n + 1)
    vol = np.zeros(n + 1)

    X[0] = X_0
    vol[0] = vol_0
    for i, dB_1_t, dB_2_t, t in zip(range(1, n+1), dB_1, dB_2, partition):
        # 1つ前の X と sig の値
        pre_vol = vol[i-1]
        pre_X = X[i-1] 
        # X と sig の値を計算（SDEに従う）
        X[i] = pre_X + np.sqrt(pre_vol) * pre_X**(beta) * dB_1_t
        vol[i] = pre_vol + alpha * pre_vol * (rho * dB_1_t + np.sqrt(1 - rho**2) * dB_2_t)
        
    # print('sig size : {}, X size : {}, partition size : {}'.format(vol.size, X.size, len(partition)))
    
    data_array = np.array([partition, X, vol]).T
    df = pd.DataFrame(data_array, columns = ['timestamp', 'process', 'volatility'])
    
    if fig_mode:
        fig, ax = plt.subplots()

        # plot the process X and volatility sigma
        ax.plot(partition, X, color = 'blue', label = 'process')
        ax.plot(partition, vol, color = 'red', label = 'volatility')
        ax.set_xlabel('time(s)')
        ax.set_ylabel('process X and volatility vol')

        # 以下はそんなに関係ないから気にしなくていい．
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().yaxis.set_ticks_position('left')
        plt.gca().xaxis.set_ticks_position('bottom')

        plt.legend()
    
    return df

def SABR_vol(dB_1, dB_2, n, delta_t, a, rho, vol_0):
    vols_root = [vol_0]
    for i in range(1, n + 1):
        pre_vol = vols_root[i-1]
        vol_root = pre_vol * math.exp((1/2) * a * (rho * dB_1[i-1] + np.sqrt(1 - rho**2) * dB_2[i-1]) - (1/4) * (a**2) * delta_t)
        vols_root.append(vol_root)
    return vols_root

def simulate_SABR_2(n = 100, T = 1, alpha = 0.3, beta = 1, rho = 0.5, X_0 = 0.1, vol_0 = 0.1, fig_mode = False):
    # BMの生成
    BM_1, dB_1, partition = brownian_motion(n, T)
    BM_2, dB_2, partition = brownian_motion(n, T)
    dt = 1. * T / n
    
    partition = [dt*i for i in range(n + 1)]
    
    # SABR model に従う ”Process X”と ”volatility sig” を作成
    X = np.zeros(n + 1)
    X[0] = X_0
    
    SABR_vols = SABR_vol(dB_1 = dB_1, dB_2 = dB_2, n = n, delta_t = dt, a = alpha, rho = rho, vol_0 = vol_0)
    
    for i, dB_1_t, dB_2_t, t in zip(range(1, n+1), dB_1, dB_2, partition):
        # 1つ前の X と sig の値
        pre_vol = SABR_vols[i-1]
        pre_X = X[i-1] 
        # X と sig の値を計算（SDEに従う）
        X[i] = pre_X + pre_vol * pre_X**(beta) * dB_1_t
        
#     print('vol size : {}, X size : {}, partition size : {}'.format(len(SABR_vols), X.size, len(partition)))
    
    data_array = np.array([partition, X, SABR_vols]).T
    df = pd.DataFrame(data_array, columns = ['timestamp', 'process', 'volatility'])
    
    if fig_mode:
        fig, ax = plt.subplots()

        # plot the process X and volatility sigma
        ax.plot(partition, X, color = 'blue', label = 'process')
        ax.plot(partition, sig, color = 'red', label = 'volatility')
        ax.set_xlabel('time(s)')
        ax.set_ylabel('process X and volatility sigma')

        # 以下はそんなに関係ないから気にしなくていい．
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().yaxis.set_ticks_position('left')
        plt.gca().xaxis.set_ticks_position('bottom')

        plt.legend();

    return df

#ディレクレ核
def D(N,x):
    return 1/(2*N + 1) * (np.sin(2 * np.pi * (N + 1/2) * x)) / (np.sin(2 * np.pi * x / 2) ) 

def cal_spotVol(n, N, y, K, T, alpha, beta, rho, X_0, vol_0):
    partition = [i/n for i in range(n)]
    df_path = simulate_SABR_2(n = n, T = T, alpha = alpha, beta = beta, rho = rho, X_0 = X_0, vol_0 = vol_0)
    X = df_path['process'].values
  
    vol_process = []
    #各時刻tに対して分割i,jに対するスポットボラティリティを計算する
    #nは分割数
    for t in partition:
        total_i = 0
        for i in range(K):
            add_j = 0
            for j in range(n):
                if (np.sin(2 * np.pi* (t - partition[j] + y[i]) /2)  == 0):
                    D_ = 1
                else:
                    D_ = D(N, t - partition[j] + y[i])
                
                add_j += D_ * (X[j+1] - X[j])
                
            total_i += add_j**2
    
        spot_vol = (2*N + 1) * total_i / K
        vol_process.append(spot_vol)

    #個数を合わせる為
    df_path = df_path[:-1]

    df_path['spot volatility'] = vol_process
    
    if beta != 0:
        df_path['abs_error_vol'] = np.abs((df_path['process'] * df_path['volatility'])**2 - df_path['spot volatility'])
    else:
        df_path['abs_error_vol'] = np.abs(df_path['volatility']**2 - df_path['spot volatility'])

    return df_path

#L2誤差を吐き出す関数を定義
def cal_error(df_path):
    error = np.linalg.norm(df_path['volatility']**2 - df_path['spot volatility'], ord=2) / np.sqrt(n)
    return error

def main():
    # 初期設定（適当に変更）
    N = 100
    T = 1
    alpha = 0.3
    beta = 1
    rho = -0.2
    X_0 = 0.06
    sig_0 = 0.06

    df_path = simulate_SABR(N = N, T = T, alpha = alpha, beta = beta, rho = rho, X_0 = X_0, sig_0 = sig_0, fig_mode = False)

    return df_path

if __name__ == '__main__':
    main()
