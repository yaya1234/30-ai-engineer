import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. 加载原始数据
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
print("原始数据形状:", df.shape)

# 2. 数据清洗（复制一份避免警告）
df = df.copy()

# 填充年龄中位数
df['Age'] = df['Age'].fillna(df['Age'].median())

# 填充登船港口众数
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# 3. 特征工程
# 家庭大小
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# 提取称呼（Title）
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
title_map = {'Mr':0, 'Miss':1, 'Mrs':2, 'Master':3, 'Dr':4, 'Rev':5, 'Col':6, 'Major':7, 'Mlle':8, 'Countess':9, 'Ms':10, 'Lady':11}
df['Title'] = df['Title'].map(title_map).fillna(12).astype(int)

# 4. 编码分类特征
# Sex
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Embarked
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# 5. 删除不需要的列（Name, Ticket, Cabin, PassengerId）
df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)

# 6. 确保所有列都是数值类型，并处理可能的 NaN
# 检查所有列
for col in df.columns:
    if df[col].dtype == 'object':
        print(f"警告：列 {col} 仍然是对象类型，尝试转换")
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # 填充缺失值（用中位数）
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

# 7. 选择特征
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'Title']
X = df[features]
y = df['Survived']

# 8. 最后一次检查缺失值
if X.isnull().any().any():
    print("仍有缺失值，用各列中位数填充")
    X = X.fillna(X.median())

# 9. 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 10. 训练模型
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
print("逻辑回归准确率:", accuracy_score(y_test, lr.predict(X_test)))

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
print("决策树准确率:", accuracy_score(y_test, dt.predict(X_test)))

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
print("随机森林准确率:", accuracy_score(y_test, rf.predict(X_test)))

# 11. 保存模型
joblib.dump(rf, 'titanic_rf_model.pkl')
print("模型已保存为 titanic_rf_model.pkl")