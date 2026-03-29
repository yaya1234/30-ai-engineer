import pytest
import pandas as pd
import numpy as np
from titanic_analysis import load_data, clean_data

def test_load_data():
    # 优先使用本地文件（如果存在），否则用网络 URL
    try:
        df = load_data("titanic.csv")
    except:
        df = load_data("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
    assert df.shape[0] > 0
    assert 'Survived' in df.columns

def test_clean_data():
    data = {
        'Survived': [0, 1],
        'Pclass': [3, 1],
        'Name': ['Mr. John', 'Mrs. Jane'],
        'Sex': ['male', 'female'],
        'Age': [np.nan, 30],
        'SibSp': [0, 1],
        'Parch': [0, 0],
        'Ticket': ['A', 'B'],
        'Fare': [7.25, 71.28],
        'Cabin': [np.nan, 'C85'],
        'Embarked': ['S', np.nan]
    }
    df_test = pd.DataFrame(data)
    df_clean = clean_data(df_test)
    
    assert df_clean['Age'].isnull().sum() == 0
    assert df_clean['Embarked'].isnull().sum() == 0
    assert 'FamilySize' in df_clean.columns
    assert 'Title' in df_clean.columns
    assert df_clean['Sex'].dtype in ['int64', 'float64']