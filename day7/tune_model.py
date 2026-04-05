import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import joblib

# 加载数据
df = pd.read_csv('../day2/titanic_cleaned.csv')

# 编码分类特征
sex_map = {"male": 0, "female": 1}
embarked_map = {"S": 0, "C": 1, "Q": 2}
title_map = {
    "Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Dr": 4, "Rev": 5,
    "Col": 6, "Major": 7, "Mlle": 8, "Countess": 9, "Ms": 10, "Lady": 11
}

df['Sex'] = df['Sex'].map(sex_map)
df['Embarked'] = df['Embarked'].map(embarked_map)
df['Title'] = df['Title'].map(title_map).fillna(12).astype(int)

# 特征和目标
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'Title']
X = df[features]
y = df['Survived']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 网格搜索
grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                           param_grid,
                           cv=5,
                           scoring='accuracy',
                           n_jobs=-1,
                           verbose=1)
grid_search.fit(X_train, y_train)

print("最佳参数:", grid_search.best_params_)
print("最佳交叉验证分数: {:.4f}".format(grid_search.best_score_))

# 测试集评估
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print("测试集准确率: {:.4f}".format(test_accuracy))

# 保存模型
joblib.dump(best_model, 'titanic_rf_best.pkl')
print("最佳模型已保存为 titanic_rf_best.pkl")