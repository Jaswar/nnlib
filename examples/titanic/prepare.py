import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler


def scale(X, column):
    scaler = MinMaxScaler()
    transformed = scaler.fit_transform(X[:, column].reshape(-1, 1))
    X[:, column] = transformed[:, 0]


def main(argv):
    if len(argv) < 2:
        print('Filepath to the train file needs to be specified')
        return

    filepath = argv[1]
    print(f'Processing {filepath}')

    dataset = pd.read_csv(filepath)
    dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
    dataset = pd.get_dummies(dataset, prefix=['Sex'], columns=['Sex'], drop_first=True)
    dataset = dataset.drop(['PassengerId', 'Pclass', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
    print(dataset.head())

    X = dataset.iloc[:, 1:].to_numpy()
    y = dataset.iloc[:, 0].to_numpy()
    print(X)

    scale(X, 0)  # Scale age
    scale(X, 1)  # Scale SibSp
    scale(X, 2)  # Scale Parch
    scale(X, 3)  # Scale Fare
    print(X)

    dfX = pd.DataFrame(X)
    dfy = pd.DataFrame(y)
    dfX.to_csv('./out/X.csv', header=None, index=False)
    dfy.to_csv('./out/y.csv', header=None, index=False)


if __name__ == '__main__':
    main(sys.argv)

