import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
#%%
#寫法邏輯
#IVOL_weights = IVOL_weights(rebalance_date, top_stocks_name, period_TSE_Stock_Price, period_TW50_TR)
#input 1: 調整日的日期
#input 2: input tickers need to be sorted
#input 3: 將調整日的日期往前一年作為訓練選股
#input 4: 計算這些股票的選股
#input 5: return data
#data: slice return data
#def rebalance date, tickers, return data, 
#output: 

#years = [2017, 2018, 2019, 2020, 2021]
#dt_years_first_day = [pd.to_datetime(year, format='%Y') for year in years]
#dt_years_end_day = [pd.to_datetime(year + 1, format='%Y') - DateOffset(days=1) for year in years]



def IVOL_weights2(backtest_end_day, tickers, period_TSE_TR, period_TW50_TR, top_weights):
    #backtest_first_day = backtest_end_day - DateOffset(years=1)
    data_return = pd.concat([period_TSE_TR, period_TW50_TR],axis=1)
    
    first_q_dates = data_return.groupby(data_return.index.to_period('Q')).apply(lambda x: x.index[0])
    first_q_dates = sorted(first_q_dates.tolist())

    end_q_dates = data_return.groupby(data_return.index.to_period('Q')).apply(lambda x: x.index[-1])
    end_q_dates = sorted(end_q_dates.tolist())
    
    first_m_dates = data_return.groupby(data_return.index.to_period('M')).apply(lambda x: x.index[0])
    first_m_dates = sorted(first_m_dates.tolist())
    
    end_m_dates = data_return.groupby(data_return.index.to_period('M')).apply(lambda x: x.index[-1])
    end_m_dates = sorted(end_m_dates.tolist())
    
    prev_rebalance_date = data_return.index[0]
    monthly_returns = [] #用來儲存每月的報酬
    
    for end_m_date in end_m_dates:
        
        period_return = data_return[prev_rebalance_date: end_m_date]
        period_cumulative_return = period_return.add(1).prod()-1
        monthly_returns.append(period_cumulative_return)        
        
        # Update the prev_rebalance_date for the next interval
        prev_rebalance_date = end_m_date + DateOffset(days=1)
    
    # 初始化 LinearRegression 模型
    lm = LinearRegression()
    
    # 創建一個儲存MSE的字典
    residual_dict_3M = {}
    residual_dict_6M = {}
    residual_dict_12M = {}
    
    # 假設 monthly_return 是一個包含 12 個 pandas Series 的列表
    # 每個 Series 包含多列（一列代表一個日期）和多行（一行代表一個股票）
    
    # 獲取所有的股票名稱 (排除 "TSE")
    stock_names = monthly_returns[0].drop("TW50TR_Return").index

    # 進行迴歸分析
    for stock in stock_names:
        y = np.array([monthly_return["TW50TR_Return"] for monthly_return in monthly_returns])
        X = np.array([monthly_return[stock] for monthly_return in monthly_returns]).reshape(-1, 1)
        
        # 訓練模型
        lm.fit(X, y)
        
        # 進行預測
        y_pred = lm.predict(X)
        
        # 計算MSE
        residual = y_pred - y
        
        # 將MSE儲存到字典
        residual_dict_12M[stock] = residual.std()
        
        y = y[-6:]
        X = X[-6:]
        
        # 訓練模型
        lm.fit(X, y)
        
        # 進行預測
        y_pred = lm.predict(X)
        
        # 計算MSE
        residual = y_pred - y
        
        # 將MSE儲存到字典
        residual_dict_6M[stock] = residual.std()

        y = y[-3:]
        X = X[-3:]
        
        # 訓練模型
        lm.fit(X, y)
        
        # 進行預測
        y_pred = lm.predict(X)
        
        # 計算MSE
        residual = y_pred - y
        
        # 將MSE儲存到字典
        residual_dict_3M[stock] = residual.std()

    stdev_dict_3M = {}
    stdev_dict_6M = {}
    stdev_dict_12M = {}
    
    for stock in stock_names:
        stock_return = data_return.loc[:,stock]
        quarter_stdev = stock_return[-60:].std()
        semiannual_stdev = stock_return[-120:].std()
        year_stdev = stock_return.std()
        stdev_dict_3M[stock] = quarter_stdev
        stdev_dict_6M[stock] = semiannual_stdev
        stdev_dict_12M[stock] = year_stdev
    
    
    cummulative_return_3M = data_return.loc[:,stock_names][:-60].add(1).prod() - 1
    cummulative_return_6M = data_return.loc[:,stock_names][:-120].add(1).prod() - 1
    cummulative_return_12M = data_return.loc[:,stock_names][:].add(1).prod() - 1
    
    MA60_price = data_return.loc[:,stock_names][:].add(1).cumprod()
    MA60 = MA60_price.rolling(window = 60).mean()
    MA60_result = MA60_price.iloc[-1] / MA60.iloc[-1]
    #%%
    # 將 MSE 字典轉換為 DataFrame 以便查看
    residual_dict_3M = pd.DataFrame(list(residual_dict_3M.items()), columns=['Stock', 'Residual'])
    residual_dict_6M = pd.DataFrame(list(residual_dict_6M.items()), columns=['Stock', 'Residual'])
    residual_dict_12M = pd.DataFrame(list(residual_dict_12M.items()), columns=['Stock', 'Residual'])

    #print(residual_dict)
    
    # 對 Residual 列進行排序，預設是升序
    residual_dict_3M = residual_dict_3M.sort_values(by='Residual')
    residual_dict_6M = residual_dict_6M.sort_values(by='Residual')
    residual_dict_12M = residual_dict_12M.sort_values(by='Residual')
    
    residual_dict_3M.set_index('Stock', inplace=True)
    residual_dict_6M.set_index('Stock', inplace=True)
    residual_dict_12M.set_index('Stock', inplace=True)
    
    residual_dict_3M['RANK'] = residual_dict_3M['Residual'].rank()
    residual_dict_6M['RANK'] = residual_dict_6M['Residual'].rank()
    residual_dict_12M['RANK'] = residual_dict_12M['Residual'].rank()

    residual_dict = pd.DataFrame(residual_dict_3M['RANK'] + residual_dict_6M['RANK'] + residual_dict_12M['RANK'])
#%%    
    # 將 MOM 字典轉換為 DataFrame 以便查看
    cummulative_return_3M = pd.DataFrame(list(cummulative_return_3M.items()), columns=['Stock', 'MOM'])
    cummulative_return_6M = pd.DataFrame(list(cummulative_return_6M.items()), columns=['Stock', 'MOM'])
    cummulative_return_12M = pd.DataFrame(list(cummulative_return_12M.items()), columns=['Stock', 'MOM'])

    #print(residual_dict)
    
    # 對 MOM 列進行排序，預設是升序
    cummulative_return_3M = cummulative_return_3M.sort_values(by='MOM', ascending= False)
    cummulative_return_6M = cummulative_return_6M.sort_values(by='MOM', ascending= False)
    cummulative_return_12M = cummulative_return_12M.sort_values(by='MOM', ascending= False)
    
    cummulative_return_3M.set_index('Stock', inplace=True)
    cummulative_return_6M.set_index('Stock', inplace=True)
    cummulative_return_12M.set_index('Stock', inplace=True)
    
    cummulative_return_3M['RANK'] = cummulative_return_3M['MOM'].rank(ascending = False)
    cummulative_return_6M['RANK'] = cummulative_return_6M['MOM'].rank(ascending = False)
    cummulative_return_12M['RANK'] = cummulative_return_12M['MOM'].rank(ascending = False)

    MOM_dict = pd.DataFrame(cummulative_return_3M['RANK'] + cummulative_return_6M['RANK'] + cummulative_return_12M['RANK'])
#%%
    # 均線因子
    MA60_df = pd.DataFrame(list(MA60_result.items()), columns=['Stock', 'MA'])
    
    MA60_df = MA60_df.sort_values(by='MA', ascending= False)    
    
    MA60_df.set_index('Stock', inplace=True)
    
    MA60_df['RANK'] = MA60_df['MA'].rank(ascending = False)
    
    MA_dict = pd.DataFrame(MA60_df['RANK'])

#%%    
    stdev_dict_3M = pd.DataFrame(list(stdev_dict_3M.items()), columns=['Stock', 'Stdev'])
    stdev_dict_6M = pd.DataFrame(list(stdev_dict_6M.items()), columns=['Stock', 'Stdev'])
    stdev_dict_12M = pd.DataFrame(list(stdev_dict_12M.items()), columns=['Stock', 'Stdev'])

    
    # 對 Residual 列進行排序，預設是升序
    stdev_dict_3M = stdev_dict_3M.sort_values(by='Stdev')
    stdev_dict_6M = stdev_dict_6M.sort_values(by='Stdev')
    stdev_dict_12M = stdev_dict_12M.sort_values(by='Stdev')
    
    stdev_dict_3M.set_index('Stock', inplace=True)
    stdev_dict_6M.set_index('Stock', inplace=True)
    stdev_dict_12M.set_index('Stock', inplace=True)
    
    stdev_dict_3M['RANK'] = stdev_dict_3M['Stdev'].rank()
    stdev_dict_6M['RANK'] = stdev_dict_6M['Stdev'].rank()
    stdev_dict_12M['RANK'] = stdev_dict_12M['Stdev'].rank()

    stdev_dict = pd.DataFrame(0*stdev_dict_3M['RANK'] + 0*stdev_dict_6M['RANK'] + stdev_dict_12M['RANK'])

    
    residual_dict = stdev_dict*0 + residual_dict*0 + MOM_dict*0 + MA_dict*1
    #%%
    # 使用 qcut 將股票按照 'Residual' 列分成 5 組
    residual_dict = residual_dict.sort_values(by='RANK')
    residual_dict['Group'] = pd.qcut(residual_dict['RANK'], 5, labels=range(1, 6, 1))
    
    weights_df = pd.DataFrame()
    weights_df.index = residual_dict.index
    for i in range(1, 6, 1):
        residual_dict.loc[residual_dict['Group'] == i, 'change_weights'] = 1/len(residual_dict[residual_dict['Group'] == i])
        weights_df[f'P{i}'] = residual_dict.loc[residual_dict['Group'] == i, 'change_weights']
        weights_df[f'P{i}'] = weights_df[f'P{i}'].fillna(0)

    residual_dict['change_weights'] = residual_dict['change_weights'].fillna(0)

    return weights_df
