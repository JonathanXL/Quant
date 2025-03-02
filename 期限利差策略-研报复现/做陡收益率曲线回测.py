import pandas as pd
import numpy as np
from tqdm import tqdm

# =============================================
# 数据加载与预处理（增加久期数据）
# =============================================
# 假设数据包含收益率和久期，示例格式：
# date | 1Y_yield | 1Y_duration | 3Y_yield | 3Y_duration | ...
data = pd.read_excel('C:/Users/35343/Desktop/各期限国债到期收益率2018-2025.xlsx')
data['date'] = pd.to_datetime(data['date'])  # 将日期列转换为日期格式
data.set_index('date', inplace=True)  # 将日期列设置为索引

# 计算10Y-1Y利差（示例）
long_term = '5Y'  # 长期债券期限
short_term = '1Y'  # 短期债券期限
data['spread'] = (data[f'{long_term}_yield'] - data[f'{short_term}_yield']) * 10000  # 计算利差，单位为bp


# =============================================
# 计算滚动分位数（窗口为360天）
# =============================================
def calculate_quantile(series, window=360):
    """计算滚动分位数"""
    return series.rolling(window).apply(lambda x: (x[-1] >= x).mean(), raw=True)  # 计算当前值在滚动窗口中的分位数


data['quantile'] = calculate_quantile(data['spread'])  # 计算利差的滚动分位数


# =============================================
# 久期中性头寸计算函数
# =============================================
def calculate_dv01_neutral_ratio(long_dur, short_dur):
    """计算对冲DV01需要的头寸比例"""
    return short_dur / long_dur  # 做多短期:做空长期 = 长期久期:短期久期


# =============================================
# 做陡策略回测函数
# =============================================
def steepen_strategy_backtest(data, D1, D2):
    """
    做陡策略回测函数
    D1: 平仓分位数上界
    D2: 建仓分位数下界
    """
    positions = []  # 记录每笔交易
    current_position = None  # 当前持仓
    capital_gain = 0  # 累计收益

    for i in range(len(data)):
        row = data.iloc[i]  # 获取当前行数据

        # 获取久期数据
        long_dur = row[f'{long_term}_duration']  # 长期债券久期
        short_dur = row[f'{short_term}_duration']  # 短期债券久期
        ratio = calculate_dv01_neutral_ratio(long_dur, short_dur)  # 计算久期中性比例

        # 建仓条件：分位数<=D2
        if current_position is None and row['quantile'] <= D2:
            entry_spread = row['spread']  # 记录建仓时的利差
            current_position = {
                'entry_date': row.name,  # 建仓日期
                'entry_spread': entry_spread,  # 建仓利差
                'dv01_ratio': ratio,  # 久期中性比例
                'entry_long_yield': row[f'{long_term}_yield'],  # 建仓时长期债券收益率
                'entry_short_yield': row[f'{short_term}_yield'],  # 建仓时短期债券收益率
                'status': 'open'  # 持仓状态
            }

        # 平仓逻辑
        if current_position is not None:
            current_long_yield = row[f'{long_term}_yield']  # 当前长期债券收益率
            current_short_yield = row[f'{short_term}_yield']  # 当前短期债券收益率

            # 计算持有期收益率盈利
            long_yield_change = current_long_yield - current_position['entry_long_yield']  # 长期债券收益率变动
            short_yield_change = current_position['entry_short_yield'] - current_short_yield  # 短期债券收益率变动
            yield_profit = (long_yield_change + short_yield_change) * 10000  # 持有期收益率盈利（单位：bp）

            # 久期中性调整后的收益计算
            pnl = yield_profit  # 收益 = 持有期收益率盈利（单位：bp）

            # 止盈止损检查
            if pnl >= 5 or pnl <= -3:  # 触发止盈或止损
                capital_gain += pnl  # 更新累计收益
                current_position.update({
                    'exit_date': row.name,  # 平仓日期
                    'exit_spread': row['spread'],  # 平仓利差
                    'gain': pnl  # 本次交易收益
                })
                positions.append(current_position)  # 记录交易
                current_position = None  # 清空持仓
            elif row['quantile'] >= D1:  # 触发平仓条件
                capital_gain += pnl  # 更新累计收益
                current_position.update({
                    'exit_date': row.name,  # 平仓日期
                    'exit_spread': row['spread'],  # 平仓利差
                    'gain': pnl  # 本次交易收益
                })
                positions.append(current_position)  # 记录交易
                current_position = None  # 清空持仓

    return positions, capital_gain  # 返回交易记录和总收益


# =============================================
# 参数优化：网格搜索函数
# =============================================
def grid_search_optimize(data, strategy_type='steepen'):
    """
    参数优化主函数
    strategy_type: 'steepen'做陡策略 / 'flatten'做平策略
    """
    # 生成参数网格
    if strategy_type == 'steepen':
        # 做陡策略：D2 <=50% 且 D1 >= D2+20%
        d2_values = np.arange(1, 51) / 100  # D2取值范围：1%到50%
        d1_values = [max(d2 + 0.2, 0.2) for d2 in d2_values]  # D1 = D2 + 20%，且最小为20%
        params_grid = [(d2, min(d1, 1.0)) for d2, d1 in zip(d2_values, d1_values)]  # 生成参数组合
    else:
        # 做平策略：D1 >=50% 且 D2 <= D1-20%
        d1_values = np.arange(50, 101) / 100  # D1取值范围：50%到100%
        d2_values = [max(d1 - 0.2, 0.0) for d1 in d1_values]  # D2 = D1 - 20%，且最小为0%
        params_grid = [(d1, d2) for d1, d2 in zip(d1_values, d2_values)]  # 生成参数组合

    results = []  # 存储所有参数组合的回测结果

    # 遍历所有参数组合
    for D2, D1 in tqdm(params_grid, desc='参数搜索进度'):
        # 执行回测
        positions, total_gain = steepen_strategy_backtest(data, D1, D2)  # 调用做陡策略回测函数

        # 记录结果
        if positions:
            df = pd.DataFrame(positions)  # 将交易记录转换为DataFrame
            avg_gain = df['gain'].mean()  # 计算平均每笔收益
            win_rate = len(df[df['gain'] > 0]) / len(df)  # 计算胜率
        else:
            avg_gain = win_rate = 0  # 若无交易，收益和胜率为0

        results.append({
            'D1': D1,  # 平仓分位数
            'D2': D2,  # 建仓分位数
            'total_gain': total_gain,  # 总收益
            'avg_gain': avg_gain,  # 平均每笔收益
            'win_rate': win_rate  # 胜率
        })

    return pd.DataFrame(results)  # 返回所有参数组合的回测结果


# =============================================
# 执行参数优化
# =============================================
# 做陡策略优化（示例）
optimize_results = grid_search_optimize(data, strategy_type='steepen')

# 找到最优参数组合
best_params = optimize_results.loc[optimize_results['total_gain'].idxmax()]  # 选择总收益最大的参数组合
print(f"最优参数：D1={best_params['D1']:.2f}, D2={best_params['D2']:.2f}")
print(f"预期总收益：{best_params['total_gain']:.2f}bp")
print(f"平均每笔收益：{best_params['avg_gain']:.2f}bp")
print(f"胜率：{best_params['win_rate']:.2%}")

# =============================================
# 使用最优参数执行回测
# =============================================
best_D1 = best_params['D1']  # 最优平仓分位数
best_D2 = best_params['D2']  # 最优建仓分位数
final_positions, final_gain = steepen_strategy_backtest(data, best_D1, best_D2)  # 执行回测

# 输出最终结果
if final_positions:
    print(f"最终总收益：{final_gain:.2f}bp")
    print(f"交易次数：{len(final_positions)}")
else:
    print("未触发任何交易")

# =============================================
# 保存结果到Excel文件
# =============================================

# 保存最终交易记录
if final_positions:
    final_positions_df = pd.DataFrame(final_positions)
    final_positions_df.to_excel('C:/Users/35343/Desktop/final_positions.xlsx', index=False)