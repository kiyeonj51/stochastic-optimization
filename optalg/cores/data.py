import zipfile
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.preprocessing import OneHotEncoder
import os
import requests
from keras.datasets import mnist





class DataNews(object):
    @staticmethod
    def load_data():
        path = '../../dataset/news.zip'
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
        path = '../../dataset/digits.zip'
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

class DataMNIST(object):
    @staticmethod
    def load_data():
        # save mnist data
        data_path = "../data/mnist/"
        data, train, test = {}, {}, {}
        if not os.path.exists(data_path):
            os.makedirs(data_path)
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train = x_train.reshape(x_train.shape[0], -1)
            x_test = x_test.reshape(x_test.shape[0], -1)
            x_train = x_train.astype('float32') / 255
            x_test = x_test.astype('float32') / 255
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)
            data = data_dictionary(x_train, y_train, x_test, y_test)
            np.save("../data/mnist/mnist.npy",data)
        else:
            data = np.load("../data/mnist/mnist.npy", allow_pickle=True)[()]
            train = data['train']
            test = data['test']
            data = {'train':train, 'test':test}
        return data


def data_dictionary(x_train, y_train, x_test, y_test):
    data, train, test = {}, {}, {}
    train['X'] = x_train
    train['Y'] = y_train
    train['n'], train['d'] = x_train.shape
    train['C'] = len(set(train['Y']))

    test['X'] = x_test
    test['Y'] = y_test
    test['n'], test['d'] = x_test.shape
    test['C'] = len(set(test['Y']))
    data['train'] = train
    data['test'] = test
    return data
