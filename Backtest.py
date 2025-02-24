import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 绩效评价模块，计算回测指标
class Performance:
    """
    绩效评价模块，计算常见的回测指标：
    - 年化收益率
    - 最大回撤
    - 夏普比率等
    """

    @staticmethod
    def annual_return(value_series: pd.DataFrame) -> float:
        """
        计算年化收益率：假设数据是按日频率来统计。
        如果频率不同，需要根据实际情况调整。
        """
        if len(value_series) < 2:
            return 0.0
        
        start_val = value_series['total_value'].iloc[0]
        end_val = value_series['total_value'].iloc[-1]
        
        # 防止开始或结束资金为零
        if start_val == 0 or end_val == 0:
            return 0.0
        
        # 回测天数
        days = (value_series['date'].iloc[-1] - value_series['date'].iloc[0]).days
        if days == 0:
            return 0.0

        total_return = (end_val - start_val) / start_val
        annual_return = (1 + total_return) ** (365.0 / days) - 1
        return annual_return

    @staticmethod
    def max_drawdown(value_series: pd.DataFrame) -> float:
        """
        计算最大回撤
        """
        cumulative = value_series['total_value'].cummax()
        drawdown = (value_series['total_value'] - cumulative) / cumulative
        max_dd = drawdown.min()
        return max_dd

    @staticmethod
    def sharpe_ratio(value_series: pd.DataFrame, rf: float = 0.0) -> float:
        """
        计算夏普比率:
        - rf 为无风险利率，默认为0
        - 使用简单的日度收益率计算，也可使用对数收益。
        """
        value_series['daily_return'] = value_series['total_value'].pct_change().fillna(0.0)
        excess_return = value_series['daily_return'] - rf / 252  # 年化转为日化
        if excess_return.std() == 0:
            return 0.0
        sr = np.sqrt(252) * excess_return.mean() / excess_return.std()
        return sr

# 数据加载模块
class DataHandler:
    def __init__(self, file_path: str = None):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        if not self.file_path:
            raise ValueError("请指定CSV文件路径。")
        self.data = pd.read_csv(self.file_path, parse_dates=['date'])
        self.data.sort_values(by='date', inplace=True)
        self.data.reset_index(drop=True, inplace=True)

    def get_data(self):
        if self.data is None:
            raise ValueError("请先调用load_data方法加载数据。")
        return self.data

class IndexDataHandler:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.index_data = None

    def load_index_data(self):
        if not self.file_path:
            raise ValueError("请指定沪深300指数数据路径。")
        self.index_data = pd.read_csv(self.file_path, parse_dates=['date'])
        self.index_data.rename(columns={'close': 'index_close'}, inplace=True)  # 重命名为 index_close
        self.index_data.sort_values(by='date', inplace=True)
        self.index_data.reset_index(drop=True, inplace=True)

    def get_index_data(self):
        if self.index_data is None:
            raise ValueError("请先调用load_index_data方法加载数据。")
        return self.index_data

# 策略模块
class Strategy:
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("请在子类中实现该方法。")

class MovingAverageStrategy(Strategy):
    def __init__(self, short_window: int = 10, long_window: int = 30):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['ma_short'] = df['close'].rolling(window=self.short_window).mean()
        df['ma_long'] = df['close'].rolling(window=self.long_window).mean()

        df['signal'] = np.where(df['ma_short'] > df['ma_long'], 1, 0)

        df.dropna(inplace=True)
        signals = df[['date', 'signal']].copy()

        return signals

# 经纪人模拟模块
class BrokerSimulator:
    def __init__(self, slippage: float = 0.0, commission_rate: float = 0.0):
        self.slippage = slippage
        self.commission_rate = commission_rate

    def execute_order(self, date: pd.Timestamp, signal: int, price: float, shares: int = 1):
        if signal == 0:
            return None
        fill_price = price + self.slippage if signal == 1 else price - self.slippage
        commission = fill_price * shares * self.commission_rate

        trade_record = {
            "date": date,
            "signal": signal,
            "fill_price": fill_price,
            "shares": shares,
            "commission": commission
        }
        return trade_record

# 投资组合模块
class Portfolio:
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_cash = initial_capital
        self.shares = 0
        self.trade_history = []
        self.portfolio_value_series = []

    def update_after_trade(self, trade_record: dict):
        if not trade_record:
            return
        if trade_record['signal'] == 1:
            cost = trade_record['fill_price'] * trade_record['shares'] + trade_record['commission']
            self.current_cash -= cost
            self.shares += trade_record['shares']
        elif trade_record['signal'] == -1:
            revenue = trade_record['fill_price'] * trade_record['shares'] - trade_record['commission']
            self.current_cash += revenue
            self.shares -= trade_record['shares']

        self.trade_history.append(trade_record)

    def update_portfolio_value(self, date, price):
        market_value = self.shares * price
        total_value = self.current_cash + market_value
        self.portfolio_value_series.append({
            "date": date,
            "total_value": total_value,
            "cash": self.current_cash,
            "shares": self.shares
        })

    def get_value_series_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.portfolio_value_series)

# 回测引擎
class BacktestingEngine:
    def __init__(self, data_handler, strategy, broker, portfolio, index_data_handler):
        self.data_handler = data_handler
        self.strategy = strategy
        self.broker = broker
        self.portfolio = portfolio
        self.index_data_handler = index_data_handler

    def run_backtest(self):
        data = self.data_handler.get_data()
        index_data = self.index_data_handler.get_index_data()
        signals = self.strategy.generate_signals(data)

        merged_data = pd.merge(data, signals, on='date', how='left')
        merged_data = pd.merge(merged_data, index_data[['date', 'index_close']], on='date', how='left')
        merged_data['signal'] = merged_data['signal'].fillna(0)

        self.portfolio_value_series = []
        for i in range(1, len(merged_data)):
            current_date = merged_data.iloc[i]['date']
            current_price = merged_data.iloc[i]['close']
            previous_price = merged_data.iloc[i-1]['close']

            if previous_price == 0:
                merged_data.loc[i, 'index_return'] = 0
            else:
                merged_data.loc[i, 'index_return'] = (current_price / previous_price) - 1

            current_signal = merged_data.iloc[i]['signal']
            if current_signal == 1:
                trade_record = self.broker.execute_order(current_date, 1, current_price, shares=100)
                self.portfolio.update_after_trade(trade_record)
            elif current_signal == 0 and merged_data.iloc[i-1]['signal'] == 1:
                trade_record = self.broker.execute_order(current_date, -1, current_price, shares=100)
                self.portfolio.update_after_trade(trade_record)

            self.portfolio.update_portfolio_value(current_date, current_price)

        value_series = self.portfolio.get_value_series_df()
        value_series['strategy_return'] = value_series['total_value'].pct_change().fillna(0)

        merged_data['strategy_return'] = value_series['strategy_return']
        merged_data['excess_return'] = merged_data['strategy_return'] - merged_data['index_return']

        return merged_data, value_series

    def output_performance(self, merged_data, value_series):
        ann_ret = Performance.annual_return(value_series)
        max_dd = Performance.max_drawdown(value_series)
        sharpe = Performance.sharpe_ratio(value_series, rf=0.02)

        print("============== 回测绩效 ==============")
        print(f"年化收益率: {ann_ret * 100:.2f}%")
        print(f"最大回撤: {max_dd * 100:.2f}%")
        print(f"夏普比率: {sharpe:.2f}")
        print("=====================================")

        self.plot_performance(merged_data)

    def plot_performance(self, merged_data):
        plt.figure(figsize=(12, 6))
        plt.plot(merged_data['date'], merged_data['strategy_return'].cumsum(), label="策略累计收益")
        plt.plot(merged_data['date'], merged_data['index_return'].cumsum(), label="沪深300基准累计收益")
        plt.plot(merged_data['date'], merged_data['excess_return'].cumsum(), label="超额收益")
        plt.xlabel('日期')
        plt.ylabel('累计收益')
        plt.title('策略与基准收益对比')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# 主函数
if __name__ == "__main__":
    file_path = "path_to_your_stock_data.csv"
    index_file_path = "path_to_your_index_data.csv"

    # 数据加载
    data_handler = DataHandler(file_path=file_path)
    data_handler.load_data()

    index_data_handler = IndexDataHandler(file_path=index_file_path)
    index_data_handler.load_index_data()

    # 配置策略
    ma_strategy = MovingAverageStrategy(short_window=10, long_window=30)

    # 配置模拟交易撮合
    broker = BrokerSimulator(slippage=0.01, commission_rate=0.001)

    # 配置投资组合
    portfolio = Portfolio(initial_capital=100000.0)

    # 初始化回测引擎
    engine = BacktestingEngine(data_handler, ma_strategy, broker, portfolio, index_data_handler)
    merged_data, value_series = engine.run_backtest()

    # 输出回测绩效并生成图表
    engine.output_performance(merged_data, value_series)
