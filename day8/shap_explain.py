import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# 加载清洗后的数据（使用原始 URL 清洗，或直接使用 day2 的清洗文件）
from sklearn.model_selection import train_test_split

# 加载数据（与 tune_model.py 中相同的方式）
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 清洗（与之前一致）
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
title_map = {'Mr':0, 'Miss':1, 'Mrs':2, 'Master':3, 'Dr':4, 'Rev':5, 'Col':6, 'Major':7, 'Mlle':8, 'Countess':9, 'Ms':10, 'Lady':11}
df['Title'] = df['Title'].map(title_map).fillna(12).astype(int)
df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'Title']
X = df[features]
y = df['Survived']

# 加载优化后的模型
model = joblib.load('../day7/titanic_rf_best.pkl')   # 或绝对路径

# 使用 SHAP 解释器（TreeExplainer 适用于树模型）
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# 全局特征重要性（SHAP 平均绝对值）
shap.summary_plot(shap_values, X, plot_type="bar", feature_names=features)
plt.savefig('shap_feature_importance.png')
plt.show()

# 对单个样本解释（例如第1个乘客）
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:], matplotlib=True)
plt.savefig('shap_force_plot.png')
plt.show()

print("SHAP 分析完成，图表已保存。")