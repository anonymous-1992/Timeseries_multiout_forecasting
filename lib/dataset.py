import numpy as np
import pandas as pd
import os


class DataUtils:

    def __init__(self, params):

        self.window = params.window
        self.horizon = params.horizon
        self.data_dir = params.data_dir
        self.train_x, self.train_y = self.get_samples('train')

        self.val_x, self.val_y = self.get_samples('validation')

        self.test_x, self.test_y = self.get_samples('test')

    def get_samples(self, type):

        data_path = os.path.join(self.data_dir, '{}.csv'.format(type))
        data = pd.read_csv(data_path)
        data = data['SpConductivity'].values

        X, y = list(), list()
        in_start = 0
        for _ in range(len(data)):
            in_end = in_start + self.window
            out_end = in_end + self.horizon
            if out_end <= len(data):
                X.append(data[in_start:in_end])
                y.append(data[in_end:out_end])
            in_start += 1
        X = np.array(X)
        y = np.array(y)

        return (X, y)


class DataOneD:

    def __init__(self, params):

        self.data_dir = params.data_dir

        self.train = self.one_d_samples('train')

        self.validation = self.one_d_samples('validation')

        self.test = self.one_d_samples('test')

    def one_d_samples(self, type):

        data_path = os.path.join(self.data_dir, '{}.csv'.format(type))
        data = pd.read_csv(data_path)
        data = data['SpConductivity'].values
        return data