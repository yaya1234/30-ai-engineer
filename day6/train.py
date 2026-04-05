import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('titanic_cleaned.csv')
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'Title']
X = df[features]
y = df['Survived']
model = RandomForestClassifier(random_state=42)
model.fit(X, y)
joblib.dump(model, 'titanic_rf_model.pkl')
print("Model retrained successfully!")