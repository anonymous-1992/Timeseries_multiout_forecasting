import argparse
import tensorflow as tf
import pickle
import numpy as np
from eval_metrics import EvalMetrics
from dataset import DataUtils


class DeepReg:

    def __init__(self, params):

        self.train_x, self.train_y = Data.train_x, Data.train_y
        self.deep_train_x = self.train_x.reshape(self.train_x.shape[0], self.train_x.shape[1], 1)

        self.test_x, self.test_y = Data.test_x, Data.test_y
        self.deep_test_x = self.test_x.reshape(self.test_x.shape[0], self.test_x.shape[1], 1)

        self.trn_sh_1 = self.train_x.shape[0]
        self.trn_sh_2 = self.train_x.shape[1]

        self.tst_sh_1 = self.test_x.shape[0]
        self.tst_sh_2 = self.test_x.shape[1]

        self.LSTM = tf.keras.models.load_model(params.LSTM)
        self.BiLSTM = tf.keras.models.load_model(params.BiLSTM)
        self.EdLSTM = tf.keras.models.load_model(params.EdLSTM)
        self.BiEdLSTM = tf.keras.models.load_model(params.BiEdLSTM)
        self.CNN = tf.keras.models.load_model(params.CNN)
        self.GRU = tf.keras.models.load_model(params.GRU)
        self.BiGRU = tf.keras.models.load_model(params.BiGRU)
        with open(params.LR, 'rb') as file: self.LR = pickle.load(file)
        with open(params.SVR, 'rb') as file: self.SVR = pickle.load(file)
        with open(params.Lasso, 'rb') as file: self.Lasso = pickle.load(file)
        with open(params.GP, 'rb') as file: self.GP = pickle.load(file)

        self.LSTM_pred, self.BiLSTM_pred, self.EdLSTM_pred, self.BiEdLSTM_pred,\
        self.CNN_pred, self.GRU_pred, self.BiGRU_pred, self.LR_pred, self.SVR_pred,\
        self.Lasso_pred, self.GP_pred = None, None, None, None,\
                            None, None, None, None, None, None, None

    def set_prediction(self, set_type):

        deep_x, x = (self.deep_train_x, self.train_x) if set_type == 'train' \
            else (self.deep_test_x, self.test_x)

        sh_1, sh_2 = (self.train_x.shape[0], self.train_x.shape[1]) if set_type == 'train' \
            else (self.test_x[0], self.test_x[1])

        self.LSTM_pred = np.array(self.LSTM.predict(deep_x)).reshape(sh_1 * sh_2, )
        self.BiLSTM_pred = np.array(self.BiLSTM.predict(deep_x)).reshape(sh_1 * sh_2, )
        self.EdLSTM_pred = np.array(self.EdLSTM.predict(deep_x)).reshape(sh_1 * sh_2, )
        self.BiEdLSTM_pred = np.array(self.BiEdLSTM.predict(deep_x)).reshape(sh_1 * sh_2, )
        self.CNN_pred = np.array(self.CNN.predict(deep_x)).reshape(sh_1 * sh_2, )
        self.GRU_pred = np.array(self.GRU.predict(deep_x)).reshape(sh_1 * sh_2, )
        self.BiGRU_pred = np.array(self.BiGRU.predict(deep_x)).reshape(sh_1 * sh_2, )

        self.LR_pred = np.array(self.LR.predict(x)).reshape(sh_1 * sh_2, )
        self.SVR_pred = np.array(self.SVR.predict(x)).reshape(sh_1 * sh_2, )
        self.Lasso_pred = np.array(self.Lasso.predict(x)).reshape(sh_1 * sh_2, )
        self.GP_pred = np.array(self.GP.predict(x)).reshape(sh_1 * sh_2, )

    def combine(self):

        self.set_prediction('train')
        A = np.vstack((self.LSTM_pred,
                       self.BiLSTM_pred,
                       self.EdLSTM_pred,
                       self.BiEdLSTM_pred,
                       self.CNN_pred,
                       self.GRU_pred,
                       self.BiGRU_pred,
                       self.LR_pred,
                       self.SVR_pred,
                       self.Lasso_pred,
                       self.GP_pred)).T

        b = self.train_y.reshape(self.train_y.shape[0] * self.train_y.shape[1], )

        m = np.linalg.lstsq(A, b, rcond=None)[0]

        self.set_prediction('test')
        A = np.vstack((self.LSTM_pred,
                       self.BiLSTM_pred,
                       self.EdLSTM_pred,
                       self.BiEdLSTM_pred,
                       self.CNN_pred,
                       self.GRU_pred,
                       self.BiGRU_pred,
                       self.LR_pred,
                       self.SVR_pred,
                       self.Lasso_pred,
                       self.GP_pred)).T

        final_pred = np.dot(m, A)

        labels = self.test_y.reshape(self.test_y.shape[0] * self.test_y.shape[1], )

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





