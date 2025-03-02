import pandas as pd

# 读取Excel文件
file_path = 'C:/Users/35343/Desktop/各期限国债到期收益率2018-2025.xlsx'  # 替换为你的Excel文件路径
df = pd.read_excel(file_path)

# 提取6个期限的yield数据
yields = df[['6M_yield', '1Y_yield', '3Y_yield', '5Y_yield', '10Y_yield', '30Y_yield']]

# 计算15组期限利差
spreads = pd.DataFrame()

# 计算利差
spreads['1Y-6M'] = yields['1Y_yield'] - yields['6M_yield']
spreads['3Y-6M'] = yields['3Y_yield'] - yields['6M_yield']
spreads['5Y-6M'] = yields['5Y_yield'] - yields['6M_yield']
spreads['10Y-6M'] = yields['10Y_yield'] - yields['6M_yield']
spreads['30Y-6M'] = yields['30Y_yield'] - yields['6M_yield']
spreads['3Y-1Y'] = yields['3Y_yield'] - yields['1Y_yield']
spreads['5Y-1Y'] = yields['5Y_yield'] - yields['1Y_yield']
spreads['10Y-1Y'] = yields['10Y_yield'] - yields['1Y_yield']
spreads['30Y-1Y'] = yields['30Y_yield'] - yields['1Y_yield']
spreads['5Y-3Y'] = yields['5Y_yield'] - yields['3Y_yield']
spreads['10Y-3Y'] = yields['10Y_yield'] - yields['3Y_yield']
spreads['30Y-3Y'] = yields['30Y_yield'] - yields['3Y_yield']
spreads['10Y-5Y'] = yields['10Y_yield'] - yields['5Y_yield']
spreads['30Y-5Y'] = yields['30Y_yield'] - yields['5Y_yield']
spreads['30Y-10Y'] = yields['30Y_yield'] - yields['10Y_yield']

# 描述性统计量
descriptive_stats = spreads.describe().T

# 添加偏态系数和峰度系数
from scipy.stats import skew, kurtosis

for col in spreads.columns:
    descriptive_stats.loc['偏态系数', col] = skew(spreads[col].dropna())
    descriptive_stats.loc['峰度系数', col] = kurtosis(spreads[col].dropna())

# 输出到Excel文件
output_path = 'C:/Users/35343/Desktop/期限利差描述性统计量.xlsx'  # 输出文件路径
descriptive_stats.to_excel(output_path)

print(f"描述性统计量已输出到 {output_path}")