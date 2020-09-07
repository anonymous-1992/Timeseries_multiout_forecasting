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

        self.LSTM_pred = np.array(self.LSTM.predict(deep_x))
        self.LSTM_pred = self.LSTM_pred.reshape(self.LSTM_pred.shape[0] * self.LSTM_pred.shape[1])

        self.BiLSTM_pred = np.array(self.BiLSTM.predict(deep_x))
        self.BiLSTM_pred = self.BiLSTM_pred.reshape(self.BiLSTM_pred.shape[0] * self.BiLSTM_pred.shape[1])

        self.EdLSTM_pred = np.array(self.EdLSTM.predict(deep_x))
        self.EdLSTM_pred = self.BiLSTM_pred.reshape(self.EdLSTM_pred.shape[0] * self.EdLSTM_pred.shape[1])

        self.BiEdLSTM_pred = np.array(self.BiEdLSTM.predict(deep_x))
        self.BiEdLSTM_pred = self.BiEdLSTM_pred.reshape(self.BiEdLSTM_pred.shape[0] * self.BiEdLSTM_pred.shape[1])

        self.CNN_pred = np.array(self.CNN.predict(deep_x))
        self.CNN_pred = self.CNN_pred.reshape(self.CNN_pred.shape[0] * self.CNN_pred.shape[1])

        self.GRU_pred = np.array(self.GRU.predict(deep_x))
        self.GRU_pred = self.GRU_pred.reshape(self.GRU_pred.shape[0] * self.GRU_pred.shape[1])

        self.BiGRU_pred = np.array(self.BiGRU.predict(deep_x))
        self.BiGRU_pred = self.BiGRU_pred.reshape(self.BiGRU_pred.shape[0] * self.BiGRU_pred.shape[1])

        self.LR_pred = np.array(self.LR.predict(x))
        self.LR_pred = self.LR_pred.reshape(self.LR_pred.shape[0] * self.LR_pred.shape[1], )

        self.SVR_pred = np.array(self.SVR.predict(x))
        self.SVR_pred = self.SVR_pred.reshape(self.SVR_pred.shape[0] * self.SVR_pred.shape[1])

        self.Lasso_pred = np.array(self.Lasso.predict(x))
        self.Lasso_pred = self.Lasso_pred.reshape(self.Lasso_pred.shape[0] * self.Lasso_pred.shape[1])

        self.GP_pred = np.array(self.GP.predict(x))
        self.GP_pred = self.GP_pred.reshape(self.GP_pred.shape[0] * self.GP_pred.shape[1])

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
                       self.GP_pred))

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
    parser.add_argument('--LSTM', type=str, default='LSTM.h5')
    parser.add_argument('--BiLSTM', type=str, default='BiLSTM.h5')
    parser.add_argument('--EdLSTM', type=str, default='EdLSTM.h5')
    parser.add_argument('--BiEdLSTM', type=str, default='BiEdLSTM.h5')
    parser.add_argument('--CNN', type=str, default='CNN.h5')
    parser.add_argument('--GRU', type=str, default='GRU.h5')
    parser.add_argument('--BiGRU', type=str, default='BiGRU.h5')
    parser.add_argument('--LR', type=str, default='LR.pkl')
    parser.add_argument('--SVR', type=str, default='SVR.pkl')
    parser.add_argument('--Lasso', type=str, default='Lasso.pkl')
    parser.add_argument('--GP', type=str, default='GP.pkl')
    parser.add_argument('--save', type=str, required=True)

    params = parser.parse_args()

    global Data
    Data = DataUtils(params)

    deepreg = DeepReg(params)

    rmse, rse, corr = deepreg.combine()

    with open(params.save, 'a+') as f:

        f.write(f"| combined : {params.horizon} test rmse {rmse:5.4f} , test rse {rse:5.4f} , test corr {corr:5.4f} |\n")


if __name__ == '__main__':
    main()





