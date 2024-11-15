import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def preprocess_data(train, test):
    train['Age'].fillna(train['Age'].median(), inplace=True)
    test['Age'].fillna(test['Age'].median(), inplace=True)
    train['Embarked'].fillna('S', inplace=True)
    test['Fare'].fillna(test['Fare'].median(), inplace=True)

    le = LabelEncoder()
    train['Sex'] = le.fit_transform(train['Sex'])
    test['Sex'] = le.transform(test['Sex'])
    train['Embarked'] = le.fit_transform(train['Embarked'])
    test['Embarked'] = le.transform(test['Embarked'])

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = train[features]
    y = train['Survived']
    X_test = test[features]
    return X, y, X_test

