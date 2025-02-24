import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA as arima
from statsmodels.tsa.stattools import coint
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 读取数据并进行数据清理
y = pd.read_csv(r"C:\Users\j1839\Desktop\金融计量经济学\mydata\coint_ecm.csv")
y['Date'] = pd.to_datetime(y['Date'])  # 转换为日期格式
y = y.sort_values(by='Date')  # 按日期排序


# 绘制图形
plt.figure(figsize=(10, 5), dpi=500)
plt.plot(y['Date'], y['y1'], label='y1', linestyle='-', linewidth=0.75)
plt.plot(y['Date'], y['y2'], label='y2', linestyle='--', linewidth=0.75)
plt.plot(y['Date'], y['y3'], label='y3', linestyle=':', linewidth=0.75)

# 添加标题和标签
plt.title('股价序列')
plt.xlabel('交易日期')
plt.ylabel('收益价')

# 调整图例大小并移动到右上角
plt.legend(loc='upper right', fontsize='x-small', bbox_to_anchor=(1, 1))

# 显示图像
plt.show()

# 读取文件
y = pd.read_csv(r"C:\Users\j1839\Desktop\金融计量经济学\mydata\coint_ecm.csv")
# 检验是否存在协整关系
print(coint(y.y1, y.y3,trend='c', autolag='bic'))

test = arima(y.y1, exog=y.y3, trend='c')。fit()
print(test.summary())

# 得到协整组合后，计算残差序列
ect = test.resid;y['ect'] = test.resid

y1_d = y.y1.diff(1)[1:]
y3_d = y.y3.diff(1)[1:]
ect_1 = ect.shift(1)[1:]
aligned_data = pd.concat([y1_d, y3_d, ect_1], axis=1)
aligned_data.columns = ['y1_d', 'y3_d', 'ect_1']
mdl = arima(aligned_data['y1_d'], exog=aligned_data[['y3_d', 'ect_1']], trend='n')。fit()
print(mdl.summary())
