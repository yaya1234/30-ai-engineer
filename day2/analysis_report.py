import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 数据清洗
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# 特征工程
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
title_map = {'Mr':0, 'Miss':1, 'Mrs':2, 'Master':3, 'Dr':4, 'Rev':5, 'Col':6, 'Major':7, 'Mlle':8, 'Countess':9, 'Ms':10, 'Lady':11}
df['Title'] = df['Title'].map(title_map).fillna(12)

# 可视化设置
sns.set_style('whitegrid')
plt.figure(figsize=(16, 12))

# 子图1：性别生存率
plt.subplot(2,2,1)
sns.barplot(x='Sex', y='Survived', data=df, palette='Set2')
plt.title('Survival Rate by Gender')
plt.ylabel('Survival Rate')

# 子图2：年龄分布与生存
plt.subplot(2,2,2)
sns.histplot(data=df, x='Age', hue='Survived', kde=True, bins=30, palette='Set1')
plt.title('Age Distribution by Survival')

# 子图3：舱位等级生存率
plt.subplot(2,2,3)
sns.barplot(x='Pclass', y='Survived', data=df, palette='Set3')
plt.title('Survival Rate by Pclass')

# 子图4：家庭规模生存率
plt.subplot(2,2,4)
sns.barplot(x='FamilySize', y='Survived', data=df, palette='Blues_d')
plt.title('Survival Rate by Family Size')

plt.tight_layout()
plt.savefig('titanic_eda_report.png', dpi=150)
plt.show()

# 输出相关性
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()['Survived'].sort_values(ascending=False)
print("各特征与生存率相关性:\n", corr)

# 保存清洗后数据
df.to_csv('titanic_cleaned.csv', index=False)
print("清洗后数据已保存为 titanic_cleaned.csv")