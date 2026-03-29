import pandas as pd
import numpy as np

# 加载 Titanic 数据集

url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("原始数据形状：\n", df.shape)
print("缺失值统计:\n", df.isnull().sum())

# 1. 分组聚合
grouped = df.groupby('Pclass').agg({
    'Age': 'mean',
    'Fare': 'median',
    'Survived': 'mean'
}).round(2)
print("按舱位分组统计:\n", grouped)

# 2. 添加新列：票价等级
fare_bins = pd.cut(df['Fare'], bins=3, labels=['Low', 'Medium', 'High'])
df_with_cat = df.assign(FareCategory=fare_bins)
print("票价等级分布:\n", df_with_cat['FareCategory'].value_counts())

# 3. 透视表：性别+舱位 vs 生存率
pivot = pd.pivot_table(df, values='Survived', index='Sex', columns='Pclass', aggfunc='mean')
print("透视表:\n", pivot)

# 4. 时间序列模拟
date_rng = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
ts_df = pd.DataFrame(date_rng, columns=['date'])
ts_df['value'] = np.random.randn(len(date_rng))
ts_df.set_index('date', inplace=True)
print("时间序列前5行:\n", ts_df.head())
print("按月重采样均值:\n", ts_df.resample('ME').mean().head())

# 5. 自定义函数 apply
def age_group(age):
    if pd.isna(age):
        return 'Unknown'
    elif age < 18:
        return 'Child'
    elif age < 60:
        return 'Adult'
    else:
        return 'Senior'

df['AgeGroup'] = df['Age'].apply(age_group)
print("年龄组分布:\n", df['AgeGroup'].value_counts())