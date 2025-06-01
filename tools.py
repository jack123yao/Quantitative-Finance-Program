import pandas as pd
import numpy as np
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, adfuller
import statsmodels.api as sm
from tqdm import tqdm 
import warnings
warnings.filterwarnings('ignore')

DATES_PATH = './data/dates/trading_dates.parquet'
DAILY_PATH = './data/data_daily/daily.parquet'
BASIC_INFO_PATH = './data/stock_info/stock_info.parquet'
INDUSTRY_INFO = './data/industry/sw_industry_info.parquet'
SHARE_PATH = './data/shares/shares.parquet'
    
def trading_date_offset(date:str, offset=1):
    """
    给定日期, 找到距离该日期前offset个交易日的日期(默认偏移为1)

    Args:
    - date: 给定的日期
    - offset: 距离给定日期的偏移天数(>0)

    Return:
    1. 若给定日期偏移后超出trading_dates的数据范围(2020-2025年), 返回 'Index Outside the Range'
    2. 正常情况, 返回偏移offset个交易日的日期

    """
    # 确保偏移日大于0
    if offset < 0:
        return 'Offset should be positive'
    
    trading_dates = pd.read_parquet(DATES_PATH)
    valid_dates = trading_dates[trading_dates.trade_status == 1].trade_date.values.tolist()

    # 找到给定日期在数据中的位置
    for index in range(len(valid_dates)):
        if date <= valid_dates[index]:
            date_index = index
            break
        
    # 偏移后日期不在数据中
    if date_index - offset < 0:
        return 'Index Outside the Range'
    
    return valid_dates[date_index - offset]

def trading_dates(start_date:str, end_date:str):
    """
    给定起止日期，返回中间的所有的交易日列表(包含端点)
    
    Args:

    - start_date
    - end_date
        
    Return:
    
    - List
    """
    if start_date > end_date:
        return None
    
    trading_dates = pd.read_parquet(DATES_PATH)
    valid_dates = trading_dates[trading_dates.trade_status == 1].trade_date.values.tolist()
    
     # 找到给定日期在数据中的位置
    for index in range(len(valid_dates)):
        if start_date <= valid_dates[index]:
            start_index = index
            break
        
    for index in range(len(valid_dates)):
        if end_date < valid_dates[index]:
            end_index = index
            break
    
    return valid_dates[start_index:end_index]

def get_daily_data(start_date:str, end_date:str):
    """
    获取给定日期区间的股票日线数据(剔除上市不满一年、涨跌停的股票)
    
    Args:
    - start_date
    - end_date
    
    Return:
    - pd.DataFrame(Columns: stock_code, date, open, close, high, low, volume, money, mcap, prev_close)
    
    """
    if start_date > end_date:
        return None
    
    def filter_stocks(df:pd.DataFrame):
        """
        剔除上市不满一年的股票, 剔除涨跌停股票
        
        Args:
        - df: pd.DataFrame(Columns: stock_code,exchange,date,open,close,high,low,volume,money)
                
        Return:
        - pd.DataFrame(Columns: stock_code,exchange,date,open,close,high,low,volume,money)
        """
        stock_info = pd.read_parquet(BASIC_INFO_PATH)

        # 转换成日期格式
        df.date = pd.to_datetime(df.date)
        stock_info.list_date = pd.to_datetime(stock_info.list_date)

        # 计算上市天数
        merge_df = pd.merge(df, stock_info, on = ['stock_code'],how = 'left')
        merge_df['days_listed'] = (merge_df['date'] - merge_df['list_date']).dt.days

        # 剔除上市不满365天
        df_filtered = merge_df[merge_df['days_listed'] >= 365]

        # 剔除涨跌停股票
        df_filtered = df_filtered[df_filtered.limit_status == 0]
        return df_filtered[['stock_code','exchange','date','open','close','high','low','volume','money','mcap','prev_close']]

    result = pd.read_parquet(DAILY_PATH)
    result = result[(result.date >= start_date) & (result.date <= end_date)].reset_index(drop = True)
    return filter_stocks(result)

def get_shares(start_date:str, end_date:str):
    """
    获取指定时间短内所有A股流通股本数据
    
    Args:
   
    - start_date
    - end_date
    
    Return:
    - pd.DataFrame(包含stock_code, date, shares)
    
    """
    result = pd.read_parquet(SHARE_PATH)
    return result[(result.date >= start_date) & (result.date <= end_date)].reset_index(drop=True)

def Factor_Analysis(df, factor_name):

    def factor_mad_cut_extreme(df:pd.DataFrame, factor_name="factor_value", k=3):
        """
        因子横截面MAD去极值模块
        
        MAD:计算因子值偏离中位数的绝对偏差的中位数，作为衡量离散程度的指标
        因子值被限制在 [median(X) - k * MAD, median(X) + k * MAD]
        
        Args:
        - df: pd.Dataframe(Columns: stock_code, date, factor_value)  
        - k: 默认为3
            
        Return:
        - pd.DataFrame(Columns: stock_code, date, factor_value)

        """
        def mad_winsorize_group(group, k=k):
            median_val = group.median()
            mad = np.median(np.abs(group - median_val))
            if mad == 0:
                return group
            return np.clip(group, median_val - k * mad, median_val + k * mad)
                        
          
            
        print('正在进行MAD去极值...')
        df[factor_name] = df.groupby('date')[factor_name].transform(lambda x:mad_winsorize_group(x))

        sns.kdeplot(df[factor_name], 
                fill=True, 
                color="#82B0D2", 
                bw_adjust=0.5, 
                linewidth=2.5, 
                alpha=0.7)  # 指定第二个子图
            
        # 添加标题和标签
        plt.title("Factor Distribution (after MAD)", fontsize=10, weight='bold')
        plt.xlabel("Factor Value", fontsize=8)
        plt.ylabel("Density", fontsize=8)
        plt.grid(True, linestyle='--', alpha=0.6)  # 添加网格线
        
        return df

    def factor_neutralize(df: pd.DataFrame, factor_name="factor_value"):
        """
        对每个交易日的横截面数据进行市值+行业中性化处理，返回中性化后的因子残差。
        
        Args:
        - df: 输入 DataFrame, 需包含股票代码、日期、因子值
        
        Return:
        - pd.DataFrame: 中性化后的因子值，索引与原始数据一致
        """
        def cross_sectional_neutralize(df: pd.DataFrame, 
                                    factor_col: str = 'factor_value',
                                    date_col: str = 'date',
                                    code_col: str = 'stock_code',
                                    cap_col: str = 'market_cap',
                                    industry_col: str = 'industry_code',
                                    log_cap: bool = True) -> pd.DataFrame:
            df = df.copy()
            df = df.dropna(subset=[factor_col, cap_col, industry_col])  # 删除缺失值
            df[industry_col] = df[industry_col].astype(str)
            
            neutralized = []
            for date, group in tqdm(df.groupby(date_col), desc='正在进行中性化'):
                if len(group) < 2:
                    neutralized.append(pd.Series(np.nan, index=group.index))
                    continue
                
                y = group[factor_col].astype(float)
                X = group[[cap_col]]
                
                if log_cap:
                    if (X[cap_col] <= 0).any():
                        print(f"Warning: Non-positive market cap on {date}")
                        neutralized.append(pd.Series(np.nan, index=group.index))
                        continue
                    X[cap_col] = np.log(X[cap_col])
                
                # 添加行业哑变量
                if group[industry_col].nunique() > 1:
                    industry_dummies = pd.get_dummies(group[industry_col], prefix='industry', drop_first=True)
                    X = pd.concat([X, industry_dummies], axis=1)
                
                X = sm.add_constant(X)
                X = X.astype(float)
                try:
                    model = sm.OLS(y, X).fit()
                    resid = model.resid
                except Exception as e:
                    print(f"Error on {date}: {e}")
                    resid = pd.Series(np.nan, index=group.index)
                
                neutralized.append(resid)
            
            neutralized_series = pd.concat(neutralized).reindex(df.index)
            df[factor_col] = neutralized_series
            return df[[code_col, date_col, factor_col]]
        
        # 合并市值数据（假设 trading_dates 已定义）
        start_date = df['date'].min().strftime('%Y-%m-%d')
        end_date = df['date'].max().strftime('%Y-%m-%d')
        dates = trading_dates(start_date, end_date)
        result = []
        for date in dates:
            data = pd.read_parquet(f'./data/shares/shares_data/{date}.parquet')
            data = data[['inst', 'date', 'market_cap']]
            result.append(data)
        frame = pd.concat(result, axis=0, ignore_index=True)
        frame.columns = ['stock_code', 'date', 'market_cap']
        result = frame.sort_values(by=['stock_code', 'date']).reset_index(drop=True)
        
        factor_df = df.copy()
        result['date'] = pd.to_datetime(result['date'])
        factor_df['date'] = pd.to_datetime(factor_df['date'])
        merge_df = pd.merge(factor_df, result, on=['stock_code', 'date'], how='inner')
        
        # 合并行业数据
        industry_df = pd.read_parquet(INDUSTRY_INFO)[['stock_code', 'industry_code']]
        merged_df = pd.merge(merge_df, industry_df, on=['stock_code'], how='inner')
        
        # 中性化
        neutralized_df = cross_sectional_neutralize(merged_df, factor_col = factor_name)
        
        return neutralized_df

    def factor_zscore(df:pd.DataFrame, fill_method='zero', factor_name="factor_value"):
        """
        因子横截面 Z-Score 标准化
        
        Args:
        - df : DataFrame
            必须包含 ['stock_code', 'date', 'factor_value'] 三列
            
        fill_method : str, 可选 ('zero', 'nan', 'remove')
            处理标准差为0时的策略：
            - 'zero'：将 zscore 设为0（默认）
            - 'nan' ：保留原始 NaN
            - 'remove'：删除该日期数据
            
        Return:
        - pd.DataFrame -['stock_code', 'date', factor_name] 三列
        """
        
        print('正在进行标准化...')
        # 校验输入列
        required_cols = {'stock_code', 'date', factor_name}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")
        
        df = df.copy()
        
        # 按日期分组计算统计量
        grouped = df.groupby('date')[factor_name]
        mean = grouped.transform('mean')
        std = grouped.transform('std')
        
        # 处理零标准差
        if fill_method == 'zero':
            std = std.replace(0, 1)  # 当std=0时，分子为0，结果强制为0
        elif fill_method == 'nan':
            std = std.replace(0, np.nan)
        elif fill_method == 'remove':
            valid_dates = grouped.filter(lambda x: x.std() != 0)['date']
            df = df[df['date'].isin(valid_dates)]
            mean = df.groupby('date')[factor_name].transform('mean')
            std = df.groupby('date')[factor_name].transform('std')
        else:
            raise ValueError(f"Invalid fill_method: {fill_method}. Use 'zero', 'nan', or 'remove'")
        
        # 计算 Z-Score
        df[factor_name] = (df[factor_name] - mean) / std
        
        return df[['stock_code', 'date', factor_name]]
    
    
    return factor_zscore(factor_neutralize(factor_mad_cut_extreme(df, factor_name), factor_name), factor_name=factor_name)

def factor_rank_autocorrelation(df:pd.DataFrame):
    """
    因子自相关性分析模块
    
    Args:
    - df: pd.DataFrame()(columns: stock_code, date, factor_value)
        
    Return:
    - 中间输出: 通过平稳性检验的股票的个数
    - pd.DataFrame(): 滞后为1天、5天、10天的自相关系数
        
    """
    
    """ 计算平均因子rank自相关结果 """
    
    print('正在进行自相关性分析...')
    df['date'] = pd.to_datetime(df['date'])

    # 确保数据按日期排序
    df = df.sort_values('date')

    # 计算每日横截面因子值的rank
    df['factor_rank'] = df.groupby('date')['factor_value'].rank(method='average')

    # 对每只股票的rank序列计算自相关(n=1,5,10)
    def safe_acf(series, nlags=1):
        if len(series) < 3:  # 最少需要3个点计算可靠自相关
            return np.nan
        try:
            return acf(series, nlags=nlags, fft=False)[nlags]  
        except:
            return np.nan
        
    acf_by_stock_1 = df.groupby('stock_code')['factor_rank'].apply(safe_acf, nlags=1)
    acf_by_stock_5 = df.groupby('stock_code')['factor_rank'].apply(safe_acf, nlags=5)
    acf_by_stock_10 = df.groupby('stock_code')['factor_rank'].apply(safe_acf, nlags=10)
    
    # 计算平均自相关（忽略NaN）
    mean_acf_rank_1 = acf_by_stock_1.dropna().mean()
    mean_acf_rank_5 = acf_by_stock_5.dropna().mean()
    mean_acf_rank_10 = acf_by_stock_10.dropna().mean()

    result = pd.DataFrame(index =['Mean Factor Rank Autocorrelation'], columns = ['1D','5D','10D'])
    result['1D'] = mean_acf_rank_1.round(3)
    result['5D'] = mean_acf_rank_5.round(3)
    result['10D'] = mean_acf_rank_10.round(3)
    
    """ ADF检验 """
    
    # 检查NaN值
    nan_count = df['factor_value'].isna().sum()
    if nan_count > 0:
        print("存在NaN值，建议处理（删除或填充）。当前代码将跳过含NaN的股票。")

    df['factor_value'] = df['factor_value'].fillna(0)
    
    # 定义ADF检验函数
    def run_adf(series, min_length=3):
        
        # 检查序列长度和是否为常数
        if len(series) < min_length or series.isna().any():
            return np.nan
        if series.nunique() == 1:  # 常数序列
            return np.nan
        
        # 运行ADF检验
        result = adfuller(series)
        return result[1]

    # 对每只股票的因子值序列进行ADF检验
    adf_results = df.groupby('stock_code')['factor_value'].apply(run_adf).reset_index()

    adf_results.columns = ['stock_code','p_value']
    adf_results['stationary'] = adf_results['p_value'].apply(
        lambda p: '平稳 (p < 0.05)' if p < 0.05 else '可能非平稳 (p ≥ 0.05)' if pd.notna(p) else '无法检验'
    )
    
    # 计算平稳股票比例
    valid_results = adf_results.dropna(subset=['p_value'])
    stationary_count = (valid_results['p_value'] < 0.05).sum()
    total_valid = len(valid_results)
    print(f"\n平稳股票数量: {stationary_count}/{total_valid} ({stationary_count/total_valid:.2%})")
    return result

def factor_backtest(df:pd.DataFrame, factor_name:str, start_date:str, end_date:str, lag_days:int=2, direction = 1, group:int=5, plot:bool=True):
    """
    因子分组回测函数
    
    Args:
    - df: pd.DataFrame, 包含 stock_code, date, factor_value 三列
    - group: 分组数量，默认 10
    - lag_days: 使用滞后多少天的收益率
    - plot: 是否绘制累计收益率曲线，默认 True
            
    Return:
    - result: pd.DataFrame
    - cumulative_return: pd.DataFrame
    - ic: pd.Series
    """
    if factor_name not in df.columns:
        print('Factor name does not exist')
        return
    
    # 合并收益率信息
    ret = get_daily_data(start_date, end_date)[['stock_code','date','close']]
    ret['return'] = ret.groupby('stock_code')['close'].pct_change().fillna(0)
    ret = ret.drop(columns=['close'])
    ret['date'] = pd.to_datetime(ret['date'])
    merge_df = pd.merge(df, ret, on=['stock_code','date'], how='left')
    
    def backtest(df, factor_name, direction, num_groups=5, lag_days=2, plot=True):
        
        df = df.sort_values(['date', 'stock_code']).copy()

        # 计算滞后收益率
        df['return_adjusted'] = df.groupby('stock_code')['return'].shift(-lag_days)
        df = df.dropna(subset=['return_adjusted'])
        df['group'] = df.groupby('date')[factor_name].transform(
            lambda x: pd.qcut(x, num_groups,labels=False, duplicates='drop') + 1)
        
        # 计算IC
        ic = df.groupby('date').apply(
            lambda x: x[factor_name].corr(x['return_adjusted'], method='pearson')
        )

        # 处理分组可能不足的情况
        group_returns = df.groupby(['date', 'group'])['return_adjusted'].mean().unstack()
        all_groups = list(range(1, num_groups + 1))
        group_returns = group_returns.reindex(columns=all_groups).fillna(0)

        # 若IC为负则反转分组标签
        if direction < 0:
            reversed_cols = list(range(num_groups, 0, -1))
            group_returns.columns = reversed_cols
            
        # 指标计算
        result = pd.DataFrame(index=['value'])
        result['IC'] = round(ic.mean(), 3)
        result['ICIR'] = round(abs(ic.mean() / ic.std() if ic.std() != 0 else np.nan), 3)
        
        # 对冲收益计算
        group_returns['long_short'] = group_returns[num_groups] - group_returns[1]
        
        # 日期滞后调整
        cumulative_returns = group_returns.cumsum()
        cu_returns = (1 + group_returns).cumprod()
        cumulative_returns.dropna(inplace=True)

        # 年化收益率计算
        annual_return_long = group_returns[num_groups].mean() * 252  # 单利多头年化收益
        annual_return_short = group_returns[1].mean() * 252 # 单利空头年化收益
        annual_return_long_short = group_returns.long_short.mean() * 252 # 单利多空年化收益
        result['Long AR'] = round(annual_return_long, 3)
        result['Long MDD'] = round(((cu_returns[num_groups].cummax() - cu_returns[num_groups]) / cu_returns[num_groups].cummax()).max(), 3)
        result['Long Sharpe'] = round((group_returns[num_groups].mean() * 252 - 0.15) / (group_returns[num_groups].std() * np.sqrt(252)),3)
        result['Short AR'] = round(annual_return_short, 3)
        result['Short MDD'] = round(((cu_returns[1].cummax() - cu_returns[1]) / cu_returns[1].cummax()).max(), 3)
        result['Short Sharpe'] = round((group_returns[1].mean() * 252 - 0.15) / (group_returns[1].std() * np.sqrt(252)),3)
        result['LS AR'] = round(annual_return_long_short, 3)
        result['LS MDD'] = round(((cu_returns['long_short'].cummax() - cu_returns['long_short']) / cu_returns['long_short'].cummax()).max(), 3)
        result['LS Sharpe'] = round((group_returns.long_short.mean() * 252 - 0.15) / (group_returns.long_short.std() * np.sqrt(252)),3)
        
        # 绘图优化
        if plot:
            # 分组收益曲线
            plt.figure(figsize=(10, 6))  
            colors = plt.cm.coolwarm(np.linspace(0, 1, num_groups))
            
            for g in range(1, num_groups + 1):
                plt.plot(cumulative_returns.index, 
                        cumulative_returns[g], 
                        label=f'Group {g}',
                        color=colors[g-1],
                        alpha=1,
                        linewidth=1.2)
                
            x_ticks_interval = len(cumulative_returns.index) // 20
            plt.xticks(cumulative_returns.index[::x_ticks_interval], rotation=45)           
            plt.title(f"Cumulative Returns (Groups={num_groups})", 
                        fontsize=10, weight='bold', pad=15)
            plt.ylabel("Cumulative Return", fontsize=10, labelpad=10)
            plt.legend(loc='upper left', frameon=False)
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.show()
            
            # 多空对冲曲线
            plt.figure(figsize=(9, 6))  # 加宽画布适应横向布局
            plt.plot(cumulative_returns['long_short'], 
                    label='Long-Short Portfolio', 
                    color='#003153',  
                    linewidth=2.5,
                    linestyle='-',
                    marker='o', 
                    markersize=4,
                    markevery=30)  # 每30天标记一个点
            
            # 添加收益基准线
            plt.axhline(0, color='#2F4F4F', linestyle='--', linewidth=1.2, zorder=0)
            
            # 填充正负收益区域
            plt.fill_between(cumulative_returns.index, 
                            cumulative_returns['long_short'], 
                            0,
                            where=(cumulative_returns['long_short'] >= 0),
                            color='#4682B4',  # 金色填充正收益
                            alpha=0.15,
                            interpolate=True)
            plt.fill_between(cumulative_returns.index, 
                            cumulative_returns['long_short'], 
                            0,
                            where=(cumulative_returns['long_short'] < 0),
                            color='#4682B4',  # 钢蓝色填充负收益
                            alpha=0.15,
                            interpolate=True)
            
            plt.plot(cumulative_returns[num_groups], 
                     label='long portfolio', 
                     color=colors[num_groups-1], 
                     linewidth=1,
                     linestyle='-',
                     )
            
            plt.plot(cumulative_returns[1], 
                     label='short portfolio', 
                     color=colors[0], 
                     linewidth=1,
                     linestyle='-',
                    )

            # 坐标轴设置
            plt.xticks(cumulative_returns.index[::x_ticks_interval], rotation=45)           
            plt.title("Long-Short Performance", 
                        fontsize=10, weight='bold', pad=15)
            plt.ylabel("Cumulative Return", fontsize=10, labelpad=10)
            plt.legend(loc='upper left', frameon=False)
            plt.grid(True, linestyle=':', alpha=0.7)
            
            # 优化布局
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.15)  # 控制子图间距
            plt.show()
            
        def generate_ic_summary(ic_series):
            # 基础统计计算
            total = len(ic_series)
            positive = ic_series[ic_series > 0]
            negative = ic_series[ic_series <= 0]
            
            # 新增负IC期数占比
            stats_dict = {      
                'Days': total,
                'IC>0(%)': len(positive)/total,
                'IC<0(%)': len(negative)/total,  # 新增指标
            }

            # 创建横向DataFrame
            df = pd.DataFrame([stats_dict], index=['value'])
            
            # 精确格式化（保留4位小数，指定列用百分数）
            format_dict = {
                'Days': '{:.0f}',
                'IC>0(%)': '{:.2%}',
                'IC<0(%)': '{:.2%}',    # 期数占比用百分数
            }
            
            formatted_df = df.copy()
            for col in df.columns:
                if col in format_dict:
                    formatted_df[col] = df[col].map(lambda x: format_dict[col].format(x))
            
            return formatted_df.T  # 转置为横向表格

        # 生成结果
        ic_analysis = generate_ic_summary(ic).T
        result = pd.concat([result, ic_analysis], axis = 1)
        
        return result, group_returns, ic
    
    def plot_IC(ic_series: pd.Series, width = 15):
        """
        绘制IC时间序列图
        得到IC时间序列柱状图和累计IC曲线
        Args:
        - ic_series: pd.Series
        - width: int, 图的长度
        
        Return:
        - None
        """
        fig, ax1 = plt.subplots(figsize=(width, 6))

        # 绘制柱状图（主坐标轴）
        colors = ['#1f77b4' if v >= 0 else '#d62728' for v in ic_series]
        ax1.bar(ic_series.index, ic_series.values, 
                color=colors, 
                width=1)

        # 设置主坐标轴
        ax1.set_title('IC Series with Cumulative Sum', fontsize=14, pad=20)
        ax1.set_ylabel('IC Value', fontsize=10)
        ax1.axhline(0, color='black', linewidth=0.8)
        ax1.axhline(ic_series.mean(), color='black', linestyle='--', 
                label=f'Mean IC ({ic_series.mean():.2f})')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        # 创建次坐标轴
        ax2 = ax1.twinx()
        cumulative_ic = ic_series.cumsum()
        ax2.plot(ic_series.index, cumulative_ic, 
                color='#003153', 
                linewidth=2,
                label='Cumulative IC')
        ax2.set_ylabel('Cumulative IC', fontsize=10)
        ax2.grid(False)

        # 合并图例到右上角
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, 
                loc='upper right', 
                fontsize=10)

        # 设置主坐标轴刻度格式
        ax1.set_xticks(ic_series.index[::20])  
        ax1.tick_params(axis='x', rotation=45,  labelsize=8)  

        plt.tight_layout()
        plt.show()
    
    def factor_decay_analysis(df:pd.DataFrame, factor_name:str, num_groups = 5, direction=1):
        """"
        因子衰减分析(1-10 days)
        Args:
        - df: pd.DataFrame
        
        Return:
        - None
        """
        print('正在进行因子衰减分析...')
        decay_result = []
        for days in tqdm(range(1,11)):
            result, cumulative_returns, ic = backtest(df, 
                                                      factor_name = factor_name, 
                                                      num_groups = num_groups,
                                                      direction = direction,
                                                      lag_days = days, 
                                                      plot = False)
            decay_result.append(ic.mean())
            
        def plot_decay_analysis(data):
            indices = list(range(1,11))        

            # 创建柱状图
            plt.figure(figsize=(8, 3))
            bars = plt.bar(indices, data, 
                        color="#84D7F0", 
                        width=0.6,         # 柱子宽度
                        edgecolor='black')  # 边框颜色

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom')

            # 设置坐标轴
            plt.xticks(indices)  
            plt.xlabel('Days Decay', fontsize=12)
            plt.ylabel('IC Mean', fontsize=12)
            plt.title('IC Decay Analysis', fontsize=14, pad=20)

            # 添加网格线
            plt.grid(axis='y', alpha=0.4, linestyle='--')

            # 显示图形
            plt.tight_layout()
            plt.show()
        
        plot_decay_analysis(decay_result)

    def print_dict(df, returns_column='returns', key_width=10):
        df_dict = df.to_dict(orient='records')
        for i, row in enumerate(df_dict):
            for key, value in row.items():
                print(f"{str(key).ljust(key_width)}: {value}")
    
    # 生成分组回测结果
    result, group_returns, ic = backtest(merge_df, 
                                         factor_name = factor_name, 
                                         num_groups=group, 
                                         lag_days=lag_days, 
                                         direction = direction, 
                                         plot=plot)
    
    # 画出IC时间序列图和绩效结果
    if plot == True:
        plot_IC(ic)

    # 因子半衰期分析
    factor_decay_analysis(merge_df, 
                          factor_name = factor_name,
                          num_groups = group,
                          direction = direction)
    # 绩效打印
    print_dict(result)
    
    return result, group_returns, ic


    