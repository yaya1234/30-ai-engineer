"""Titanic数据集分析与清洗模块"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple


def load_data(url: str) -> pd.DataFrame:
    """从URL加载CSV数据"""
    return pd.read_csv(url)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """数据清洗：处理缺失值、特征工程等"""
    df = df.copy()
    
    # 填充年龄中位数（不使用 inplace）
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # 填充Embarked众数
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # 创建家庭规模特征
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # 提取Title（使用原始字符串避免转义警告）
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_mapping = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Dr': 4, 'Rev': 5, 'Col': 6, 'Major': 7, 'Mlle': 8, 'Countess': 9, 'Ms': 10, 'Lady': 11}
    df['Title'] = df['Title'].map(title_mapping).fillna(12)
    
    # 删除不需要的列（只删除存在的列）
    cols_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId']
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    
    # 性别编码
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    return df
    """数据清洗：处理缺失值、特征工程等"""
    df = df.copy()

    # 填充年龄中位数
    df["Age"].fillna(df["Age"].median(), inplace=True)

    # 填充Embarked众数
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    # 创建家庭规模特征
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # 提取Title
    df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
    title_mapping = {
        "Mr": 0,
        "Miss": 1,
        "Mrs": 2,
        "Master": 3,
        "Dr": 4,
        "Rev": 5,
        "Col": 6,
        "Major": 7,
        "Mlle": 8,
        "Countess": 9,
        "Ms": 10,
        "Lady": 11,
    }
    df["Title"] = df["Title"].map(title_mapping).fillna(12)

    # 删除不需要的列
    df.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1, inplace=True)

    # 性别编码
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    return df


def eda_visualization(df: pd.DataFrame, save_dir: str = "./") -> None:
    """绘制探索性数据分析图表"""
    sns.set_style("whitegrid")

    # 生存率分布
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.countplot(x="Survived", data=df)
    plt.title("Survival Count")

    plt.subplot(1, 2, 2)
    sns.barplot(x="Sex", y="Survived", data=df)
    plt.title("Survival Rate by Gender")
    plt.savefig(f"{save_dir}survival_analysis.png")
    plt.show()

    # 年龄分布
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    df["Age"].hist(bins=30)
    plt.title("Age Distribution")

    plt.subplot(1, 2, 2)
    sns.boxplot(x="Survived", y="Age", data=df)
    plt.title("Age vs Survival")
    plt.savefig(f"{save_dir}age_analysis.png")
    plt.show()


def main():
    # 如果网络不稳定，可以将url改为本地路径，例如 "titanic.csv"
    url = (
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    )

    df_raw = load_data(url)
    print(f"原始数据形状: {df_raw.shape}")

    df_clean = clean_data(df_raw)
    print(f"清洗后数据形状: {df_clean.shape}")

    print(df_clean.head())

    df_clean.to_csv("titanic_clean.csv", index=False)

    eda_visualization(df_clean)

    print("\n数据描述统计:")
    print(df_clean.describe())


if __name__ == "__main__":
    main()
