import pandas as pd
from src.Constants.Constants import ALL_FEATURES
from sklearn import preprocessing


def standardNormalization (df: pd.DataFrame, columnsNormalize = ALL_FEATURES):
    ''' Normalize given columns of the dataframe using a standard scaler '''
    X = preprocessing.StandardScaler().fit_transform( df[columnsNormalize].values.astype(float))
    df.update(pd.DataFrame(X, columns=columnsNormalize))


def miniMaxNormalization (df: pd.DataFrame, columnsNormalize = ALL_FEATURES):
    ''' Normalize given columns of the dataframe using a MinMax scaler '''
    X = preprocessing.MinMaxScaler().fit_transform( df[columnsNormalize].values.astype(float))
    df.update(pd.DataFrame(X, columns=columnsNormalize))