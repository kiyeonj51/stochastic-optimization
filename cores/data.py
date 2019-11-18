import zipfile
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.preprocessing import OneHotEncoder


class DataNews(object):
    @staticmethod
    def load_data():
        path = '../../Data/news.zip'
        data, train, test = {}, {}, {}
        with zipfile.ZipFile(path) as z:
            for filename in z.namelist():
                with z.open(filename) as f:
                    if ('mtx' in filename) and ('train' in filename):
                        train['X'] = sio.mmread(f).tocsr()
                        train['n'], train['d'] = train['X'].shape
                    elif ('mtx' in filename) and ('test' in filename):
                        test['X'] = sio.mmread(f).tocsr()
                        test['n'], test['d'] = test['X'].shape
                    elif ('csv' in filename) and ('train' in filename):
                        train['Y'] = pd.read_csv(f, sep=' ', header=None).values.astype('int').ravel()
                        train['C'] = len(set(train['Y']))
                    elif ('csv' in filename) and ('test' in filename):
                        test['Y'] = pd.read_csv(f, sep=' ', header=None).values.astype('int').ravel()
                        test['C'] = len(set(test['Y']))
                    else:
                        raise ValueError
        data['train'] = train
        data['test'] = test
        return data


class DataDigits(object):
    @staticmethod
    def load_data():
        path = '../../Data/digits.zip'
        data, train, test = {}, {}, {}
        with zipfile.ZipFile(path) as z:
            for filename in z.namelist():
                val = pd.read_csv(z.open(filename), sep=' ', header=None).values
                if 'y_digits' in filename:
                    val = val.astype('int').T[0]
                if ('train' in filename) and ('X' in filename):
                    train['X'] = val
                    train['n'], train['d'] = train['X'].shape
                elif ('train' in filename) and ('y' in filename):
                    train['Y'] = val
                    train['C'] = len(set(train['Y']))
                elif ('test' in filename) and ('X' in filename):
                    test['X'] = val
                    test['n'], test['d'] = test['X'].shape
                elif ('test' in filename) and ('y' in filename):
                    test['Y'] = val
                    test['C'] = len(set(test['Y']))
                else:
                    raise ValueError
        data['train'] = train
        data['test'] = test
        return data
