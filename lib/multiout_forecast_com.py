import argparse
import tensorflow as tf
import pickle
import numpy as np
from eval_metrics import EvalMetrics
from dataset import DataUtils


class DeepReg:

    def __init__(self, params):

        self.deep_model_path = params.deep_model_path
        self.reg_model_path = params.reg_model_path
        self.deep_model = tf.keras.models.load_model(self.deep_model_path)
        with open(self.reg_model_path, 'rb') as file:
            self.reg_model = pickle.load(file)

        self.train_x, self.train_y = Data.train_x, Data.train_y
        self.deep_train_x = self.train_x.reshape(self.train_x.shape[0], self.train_x.shape[1], 1)

        self.test_x, self.test_y = Data.test_x, Data.test_y
        self.deep_test_x = self.test_x.reshape(self.test_x.shape[0], self.test_x.shape[1], 1)

    def combine(self):

        deep_pred = self.deep_model.predict(self.deep_train_x)
        reg_pred = self.reg_model.predict(self.train_x)

        deep_pred = np.array(deep_pred)
        reg_pred = np.array(reg_pred)

        deep_pred = deep_pred.reshape(deep_pred.shape[0] * deep_pred.shape[1], )
        reg_pred = reg_pred.reshape(reg_pred.shape[0] * reg_pred.shape[1], )

        A = np.vstack((deep_pred, reg_pred)).T

        print(A.shape)

        b = self.train_y.reshape(self.train_y.shape[0] * self.train_y.shape[1], )

        print(b.shape)

        m, c = np.linalg.lstsq(A, b, rcond=None)[0]

        deep_pred = self.deep_model.predict(self.deep_test_x)
        reg_pred = self.reg_model.predict(self.test_x)

        deep_pred = np.array(deep_pred)
        reg_pred = np.array(reg_pred)

        deep_pred = deep_pred.reshape(deep_pred.shape[0] * deep_pred.shape[1], )
        reg_pred = reg_pred.reshape(reg_pred.shape[0] * reg_pred.shape[1], )

        A = np.vstack((deep_pred, reg_pred)).T

        final_pred = m * A + c

        final_pred = np.array(final_pred)

        final_pred = final_pred.reshape(final_pred.shape[0] * final_pred.shape[1], )

        out_seq = self.test_y

        labels = out_seq.reshape(out_seq.shape[0] * out_seq.shape[1], )

        eval = EvalMetrics(labels, final_pred)

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

    with open(params.save, 'a+') as f:

        f.write(f" combined : test rmse {rmse:5.4f} , test rse {rse:5.4f} , test corr {corr:5.4f} |")


if __name__ == '__main__':
    main()





