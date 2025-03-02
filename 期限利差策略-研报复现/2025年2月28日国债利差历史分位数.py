import pandas as pd
import numpy as np

# =============================================
# 数据加载与预处理
# =============================================
# 假设数据包含收益率，示例格式：
# date | 1Y_yield | 3Y_yield | 5Y_yield | ...
data = pd.read_excel('C:/Users/35343/Desktop/各期限国债到期收益率2018-2025.xlsx')
data['date'] = pd.to_datetime(data['date'])  # 将日期列转换为日期格式
data.set_index('date', inplace=True)  # 将日期列设置为索引

# 计算3Y-1Y和5Y-1Y利差
data['3Y_1Y_spread'] = (data['3Y_yield'] - data['1Y_yield']) * 10000  # 计算3Y-1Y利差，单位为bp
data['5Y_1Y_spread'] = (data['5Y_yield'] - data['1Y_yield']) * 10000  # 计算5Y-1Y利差，单位为bp


# =============================================
# 计算历史分位数
# =============================================
def calculate_quantile(series, target_date, window=360):
    """
    计算目标日期的滚动分位数
    series: 利差数据
    target_date: 目标日期
    window: 滚动窗口大小（默认360天）
    """
    # 获取目标日期之前的数据
    historical_data = series.loc[:target_date]  # 目标日期及之前的数据
    if len(historical_data) < window:
        return np.nan  # 如果数据不足，返回NaN

    # 计算滚动分位数
    rolling_quantile = historical_data.rolling(window).apply(lambda x: (x[-1] >= x).mean(), raw=True)
    return rolling_quantile.loc[target_date]  # 返回目标日期的分位数


# 目标日期
target_date = pd.to_datetime('2025-02-28')

# 计算3Y-1Y利差的分位数
quantile_3Y_1Y = calculate_quantile(data['3Y_1Y_spread'], target_date)
print(f"2025年2月28日，3Y-1Y利差的历史分位数：{quantile_3Y_1Y:.2%}")

# 计算5Y-1Y利差的分位数
quantile_5Y_1Y = calculate_quantile(data['5Y_1Y_spread'], target_date)
print(f"2025年2月28日，5Y-1Y利差的历史分位数：{quantile_5Y_1Y:.2%}")