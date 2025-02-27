
# 一、股票策略回测框架

## 简介

这是一个基于 **Python** 的股票策略回测框架，旨在帮助用户测试和评估其交易策略的表现。框架包括了数据加载、策略生成、经纪人模拟、投资组合管理和绩效评估模块，并支持沪深300指数作为基准，计算年化收益率、最大回撤、夏普比率等回测指标。

该回测框架的目标是提供一个简洁、易用的工具，帮助量化交易者快速实现和优化策略。

## 功能特点

- **策略开发与回测**：可以方便地添加和测试自己的策略，支持基于移动平均线（MA）策略的示例。
- **沪深300基准**：支持沪深300指数作为基准，计算策略的超额收益。
- **绩效评估**：计算年化收益率、最大回撤、夏普比率等常见回测指标。
- **灵活的模拟交易环境**：模拟实际的买卖过程，支持滑点和交易手续费的设置。

## 目录

- [安装](#安装)
- [使用方法](#使用方法)
- [模块说明](#模块说明)
- [贡献](#贡献)
- [许可证](#许可证)

## 安装

### 克隆仓库

```bash
git clone https://github.com/your-username/quant-backtest.git
cd quant-backtest
```
- **安装依赖**
-确保你已经安装了 Python 3.x，然后可以使用 pip 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

-**1.准备数据**：将你自己的行情数据和沪深300指数数据准备好。文件格式应该是 CSV，包含 date 和 close 列。

-**2.配置和运行回测**： 修改 main.py 文件，指定你的行情数据文件路径和沪深300指数数据路径。然后运行代码：

-**3.查看结果**：回测完成后，系统会输出回测的绩效指标（如年化收益率、最大回撤和夏普比率），并展示策略收益、基准收益和超额收益的图表。


## 模块说明

-**1.数据加载模块**：负责加载并处理市场数据和沪深300指数数据。支持通过 CSV 文件导入数据，并确保数据按日期升序排列。

-**2.策略模块**： 策略模块是回测框架的核心部分。在 Strategy 类的基础上，您可以自定义您的交易策略。示例中提供了一个基于 双均线策略（短期均线和长期均线）的实现。

-**3.经纪人模拟模块**：模拟实际交易过程，包括买入、卖出，并计算滑点和交易手续费。这个模块确保回测的交易过程尽可能地贴近实际市场。

-**4.投资组合模块**：记录持仓、现金、交易历史和总组合价值等信息，并在每次交易后更新组合状态。它还可以返回每个交易日的总价值序列，以供绩效评估使用。

-**5. 绩效评估模块**：计算并输出常见的回测指标：年化收益率、最大回撤、夏普比率等。

-**6. 回测引擎模块**：回测引擎是框架的主控制模块，负责数据加载、策略信号生成、交易模拟和绩效评估。运行回测时，它将整合所有模块并执行策略。




# 二、配对交易策略（ARIMA模型）

这是一个基于 **ARIMA** 模型的量化配对交易策略实现。该策略通过协整检验和ARIMA建模，基于两个股票（或资产）之间的价格关系进行配对交易。本项目的主要流程包括数据预处理、协整检验、策略构建、回测及策略评估。

## 项目结构

1. **数据读取与预处理**：
   - 读取CSV文件中的数据，进行时间排序和日期处理。
   
2. **协整检验**：
   - 使用 **Engle-Granger** 方法（通过 `coint` 函数）检验两只资产是否具有协整关系，并用 **ARIMA** 模型拟合协整关系。
   
3. **策略构建**：
   - 根据协整关系计算残差并构造 **z-score**。
   - 基于 z-score 构建交易信号：做多、做空或无持仓。
   
4. **策略回测与表现评估**：
   - 计算策略的收益和策略净值。
   - 评估策略表现，包括 **最大回撤** 和 **夏普比率**。

5. **回测结果展示**：
   - 绘制策略的净值曲线，帮助可视化策略的表现。

## 安装依赖

你需要安装以下Python包：
- **pandas**：数据处理
- **numpy**：数值计算
- **matplotlib**：绘图
- **statsmodels**：时间序列分析

你可以通过以下命令安装这些依赖：

```bash
pip install pandas numpy matplotlib statsmodels
```

## 数据格式

数据应包含以下列：
- **Date**：交易日期
- **y1, y2, y3**：三个资产的价格数据
- CSV文件中的数据格式示例如下：
```csv
Date,y1,y2,y3
2020-01-01,100,98,101
2020-01-02,101,99,102
...
```

## 使用方法
1. **加载数据并进行预处理**：
   ```python
   y = pd.read_csv("path_to_your_data.csv")
   y['Date'] = pd.to_datetime(y['Date'])
   y = y.sort_values(by='Date')
   ```
2. **协整检验**：
   ```python
   from statsmodels.tsa.stattools import coint
   coint_t, p_value, critical_values = coint(y.y1, y.y3, trend='c', autolag='bic')
   print("协整检验结果: coint_t={}, p_value={}, critical_values={}".format(coint_t, p_value, critical_values))
   ```
3. **ARIMA模型拟合**：
    ```python
   from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(y.y1, exog=y.y3, trend='c').fit()
    y['resid'] = model.resid
   ```
4. **构造配对交易策略**：
   ```python
   train_size = int(len(y) * 0.8)
   train_resid = y['resid'].iloc[:train_size]
   resid_mean = train_resid.mean()
   resid_std = train_resid.std()

   y['zscore'] = (y['resid'] - resid_mean) / resid_std

    # 进场和出场阈值
   entry_threshold = 2.0
   exit_threshold = 0.5

    # 交易信号
   y['signal'] = 0
   position = 0

   for i in range(train_size, len(y)):
       z = y['zscore'].iloc[i]
       if position == 0:
           if z > entry_threshold:
            position = -1  # 做空 y1，做多 y3
           elif z < -entry_threshold:
               position = 1  # 做多 y1，做空 y3
       else:
           if position == -1 and z < exit_threshold:
               position = 0  # 平仓
           elif position == 1 and z > -exit_threshold:
               position = 0  # 平仓
   
       y['signal'].iloc[i] = position

   ```
5. **计算策略收益并回测**：
   ```python
   alpha = model.params[0]
   beta = model.params[1]

   y['pos_y1'] = y['signal'] * 1.0
   y['pos_y3'] = y['signal'] * (-beta)

   y['ret_y1'] = y['pos_y1'].shift(1) * y1.pct_change()
   y['ret_y3'] = y['pos_y3'].shift(1) * y3.pct_change()
   y['strategy_ret'] = y[['ret_y1', 'ret_y3']].sum(axis=1)

   y['strategy_net'] = (1 + y['strategy_ret'].fillna(0)).cumprod()

   ```
6。 **评估策略表现**：
   ```python
   def max_drawdown(series):
       roll_max = series.cummax()
       drawdown = (series - roll_max) / roll_max
       return drawdown.min()

   def sharpe_ratio(returns, freq=252):
       mean_ret = returns.mean() * freq
       vol = returns.std() * np.sqrt(freq)
       if vol == 0:
           return 0
       return mean_ret / vol

   test_returns = test['strategy_ret']
   test_net = test['strategy_net']

   mdd = max_drawdown(test_net)
   sr = sharpe_ratio(test_returns)

   print(f"回测区间最大回撤: {mdd:.2%}")
   print(f"回测区间夏普比率: {sr:.2f}")
   ```
7。 **绘制回测净值曲线**：
   ```python
   plt.figure(figsize=(10, 5), dpi=300)
   plt.plot(test.index, test_net, label='策略净值', color='blue')
   plt.title('配对交易策略回测净值')
   plt.xlabel('Date')
   plt.ylabel('Net Value')
   plt.legend()
   plt.tight_layout()
   plt.show()
   ```
## 评估标准
1. **最大回撤（Max Drawdown）**：衡量策略在回测期间的最大损失。
2. **夏普比率（Sharpe Ratio）**：衡量策略的风险调整后回报，越高越好。


## 贡献
### 欢迎为该项目做出贡献！如果你有新的功能或修复，请通过以下步骤：

-**Fork 这个仓库**

-**创建一个新的分支**：git checkout -b feature-xyz。

-**提交你的更改**：git commit -am 'Add new feature'

-**推送到分支**：git push origin feature-xyz

-**提交一个 Pull Request**

## 许可证
### 该项目使用 MIT 许可证。详情请查看 LICENSE。
