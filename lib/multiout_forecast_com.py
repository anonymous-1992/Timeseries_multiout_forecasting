import argparse
import tensorflow as tf
import pickle
import numpy as np
import eval_metrics as eval_metrics
from dataset import DataUtils


class DeepReg:

    def __init__(self, params):

        self.deep_model_path = params.deep_model_path
        self.reg_model_path = params.reg_model_path
        self.deep_model = tf.keras.models.load_model(self.deep_model_path)
        with open(self.reg_model_path, 'rb') as file:
            self.reg_model = pickle.load(file)

        self.test_x, self.test_y = params.Data.test_x, params.Data.test_y
        self.deep_test_x = self.test_x.reshape(self.test_x[0], self.test_x[1], 1)

    def combine(self):

        deep_pred = self.deep_model.predict(self.deep_test_x)
        reg_pred = self.reg_model.predict(self.test_x)

        A = np.vstack((deep_pred, reg_pred)).T

        m, c = np.linalg.lstsq(A, self.test_y, rcond=None)[0]

        final_pred = m * A + c

        eval = eval_metrics(self.test_y, final_pred)

        rmse = eval.val_rmse()
        rse = eval.val_rse()
        corr = eval.val_corr()

        return rmse, rse, corr


def main():

    parser = argparse.ArgumentParser(description='combined Time series multi-output forecasting')
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--window', type=int, default=16)
    parser.add_argument('--horizon', type=int, default=4)
    parser.add_argument('--deep_model_path', type=str, required=True)
    parser.add_argument('--reg_model_path', type=str, required=True)
    parser.add_argument('--save', type=str, required=True)

    params = parser.parse_args()

    global Data
    Data = DataUtils(params)

    deepreg = DeepReg(params)

    rmse, rse, corr = deepreg.combine()

    print(rmse)

    with open(params.save, 'a+') as f:

        f.write(f" combined : test rmse {rmse:5.4f} , test rse {rse:5.4f} , test corr {corr:5.4f} |")








