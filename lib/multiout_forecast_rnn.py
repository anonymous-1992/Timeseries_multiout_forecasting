import argparse
import keras.callbacks as kc
from keras.layers import LSTM, GRU, Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import Bidirectional
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import tensorflow as tf
import lib.dataset as dataUtil
tf.keras.backend.set_floatx('float64')

global Data


class DeepModels:

    def __init__(self, params):

        self.params = params
        self.window = params.window
        self.horizon = params.horizon
        self.epoch_ls = params.n_epoch
        self.kernels_1_ls = params.n_kernels_1
        self.kernels_2_ls = params.n_kernels_2
        self.dropout_ls = params.dropout
        self.name = None
        self.model = Sequential()
        self.early_stop = kc.EarlyStopping(monitor='val_loss', patience=20)

    def train(self, epoch, kernel_1, kernel_2, dr):
        pass

    def evaluate(self):

        in_seq, out_seq = Data.test_x, Data.test_y

        model = tf.keras.models.load_model('{}.h5'.format(self.name))

        output = model.predict(in_seq)

        output = np.array(output)

        predictions = output.reshape(output.shape[0] * output.shape[1], )

        labels = out_seq.reshape(out_seq.shape[0] * out_seq.shape[1], )

        rmse = Data.val_rmse(labels, predictions)
        rse = Data.val_rse(labels, predictions)
        corr = Data.val_corr(labels, predictions)

        return rmse, rse, corr

    @staticmethod
    def val_rse(label, predictions):

        test_rse = 0.0
        actual_var = 0.0
        all_outputs = np.array(predictions)
        all_labels = np.array(label)
        mean_labels = all_labels.mean(axis=0)
        test_rse += np.sum((all_labels - all_outputs) ** 2)
        actual_var += np.sum((all_labels - mean_labels) ** 2)
        rse = test_rse / actual_var
        rse = float("{:.4f}".format(rse))
        return rse

    @staticmethod
    def val_rmse(label, predictions):

        mse = mean_squared_error(label, predictions)
        rmse = sqrt(mse)
        rmse = float("{:.4f}".format(rmse))
        return rmse

    @staticmethod
    def val_corr(label, predictions):

        all_outputs = np.array(predictions)
        all_labels = np.array(label)
        sigma_outputs = all_outputs.std(axis=0)
        sigma_labels = all_labels.std(axis=0)
        mean_outputs = all_outputs.mean(axis=0)
        mean_labels = all_labels.mean(axis=0)
        idx = sigma_labels != 0
        test_corr = (
                            (all_outputs - mean_outputs) * (all_labels - mean_labels)
                    ).mean(axis=0) / (sigma_outputs * sigma_labels)
        test_corr = test_corr[idx].mean()
        test_corr = float("{:.4f}".format(test_corr))
        return test_corr


class LSTMModel(DeepModels):

    def __init__(self, params):
        super(LSTMModel, self).__init__(params)
        self.name = "lstm"

    def train(self, epoch, kernels_1, kernels_2, dropout):

        self.model.add(LSTM(kernels_1, input_shape=(Data.train_x.shape[1], Data.train_x.shape[2])))
        self.model.add(Dense(kernels_2, activation='relu'))
        self.model.add(Dense(self.horizon, activation='linear'))
        self.model.add(Dropout(rate=dropout))
        self.model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        self.model.fit(Data.train_x, Data.train_y, epochs=epoch, callbacks=[self.early_stop])


class BiLSTMModel(DeepModels):

        def __init__(self, params):
            super(BiLSTMModel, self).__init__(params)

        def train(self, epoch, kernels_1, kernels_2, dropout):
            self.model.add(LSTM(kernels_1, input_shape=(Data.train_x.shape[1], Data.train_x.shape[2])))
            self.model.add(Dense(kernels_2, activation='relu'))
            self.model.add(Dense(self.horizon, activation='linear'))
            self.model.add(Dropout(rate=dropout))
            self.model.compile(loss='mse', optimizer='adam', metrics=['mse'])
            self.model.fit(Data.train_x, Data.train_y, epochs=epoch, callbacks=[self.early_stop])


class EdLSTMModel(DeepModels):

    def __init__(self, params):
        super(EdLSTMModel, self).__init__(params)

    def train(self, epoch, kernels_1, kernels_2, dropout):

        Data.train_y = Data.train_y.reshape((Data.train_y.shape[0], Data.train_y.shape[1], 1))
        self.model.add(LSTM(kernels_1, input_shape=(Data.train_x.shape[1], Data.train_x.shape[2])))
        self.model.add(RepeatVector(kernels_2))
        self.model.add(GRU(kernels_2, activation='relu', return_sequences=True))
        self.model.add(TimeDistributed(Dense(1, activation='linear')))
        self.model.add(Dropout(rate=dropout))
        self.model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        self.model.fit(Data.train_x, Data.train_y, epochs=epoch, callbacks=[self.early_stop])


class BiEdLSTMModel(DeepModels):

    def __init__(self, params):
        super(BiEdLSTMModel, self).__init__(params)

    def train(self, epoch, kernels_1, kernels_2, dropout):

        Data.train_y = Data.train_y.reshape((Data.train_y.shape[0], Data.train_y.shape[1], 1))
        self.model.add(Bidirectional(LSTM(kernels_1, input_shape=(Data.train_x.shape[1], Data.train_x.shape[2]))))
        self.model.add(RepeatVector(kernels_2))
        self.model.add(Bidirectional(LSTM(kernels_2, activation='relu', return_sequences=True)))
        self.model.add(TimeDistributed(Dense(1, activation='linear')))
        self.model.add(Dropout(rate=dropout))
        self.model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        self.model.fit(Data.train_x, Data.train_y, epochs=epoch, callbacks=[self.early_stop])


class CNNModel(DeepModels):

    def __init__(self, params):
        super(CNNModel, self).__init__(params)

    def train(self, epoch, kernels_1, kernels_2, dropout):
        self.model.add(Conv1D(kernels_1, 3, activation='relu', input_shape=(self.horizon, 1)))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(kernels_1, activation='relu'))
        self.model.add(Dense(self.horizon, activation='linear'))
        self.model.add(Dropout(rate=dropout))
        self.model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        self.model.fit(Data.train_x, Data.train_y, epochs=epoch, callbacks=[self.early_stop])


class GRUModel(DeepModels):

    def __init__(self, params):
        super(GRUModel, self).__init__(params)

    def train(self, epoch, kernels_1, kernels_2, dropout):

        self.model.add(GRU(kernels_1, input_shape=(Data.train_x.shape[1], Data.train_x.shape[2])))
        self.model.add(Dense(kernels_2, activation='relu'))
        self.model.add(Dense(self.horizon, activation='linear'))
        self.model.add(Dropout(rate=dropout))
        self.model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        self.model.fit(Data.train_x, Data.train_y, epochs=epoch, callbacks=[self.early_stop])


class BiGRUModel(DeepModels):

    def __init__(self, params):
        super(BiGRUModel, self).__init__(params)

    def train(self, epoch, kernels_1, kernels_2, dropout):

        self.model.add(Bidirectional(GRU(kernels_1, input_shape=(Data.train_x.shape[1], Data.train_x.shape[2]))))
        self.model.add(Dense(kernels_2, activation='relu'))
        self.model.add(Dense(self.horizon, activation='linear'))
        self.model.add(Dropout(rate=dropout))
        self.model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        self.model.fit(Data.train_x, Data.train_y, epochs=epoch, callbacks=[self.early_stop])


def create_model(name, params):

    if name is "LSTM":
        return LSTMModel(params)

    elif name is "BiLSTM":
        return BiLSTMModel(params)

    elif name is "EdLSTM":
        return EdLSTMModel(params)

    elif name is "BiEdLSTM":
        return BiEdLSTMModel(params)

    elif name is "CNN":
        return CNNModel(params)

    elif name is "GRU":
        return GRUModel(params)

    elif name is "BiGRU":
        return BiGRUModel(params)


def validate(params, name):

    ml = create_model(name, params)

    in_seq, out_seq = Data.val_x, Data.val_y

    labels = out_seq.reshape(out_seq.shape[0] * out_seq.shape[1], )

    best_val = float("inf")

    for epoch in ml.epoch_ls:

        for kernel_1 in ml.kernels_1_ls:

            for kernel_2 in ml.kernels_2_ls:

                for dr in ml.dropout_ls:

                    ml.train(epoch, kernel_1, kernel_2, dr)

                    output = ml.model.predict(in_seq)

                    output = np.array(output)

                    predictions = output.reshape(output.shape[0] * output.shape[1], )

                    rmse = ml.val_rmse(labels, predictions)

                    if rmse < best_val:
                        best_val = rmse
                        ml.model.save('{}.h5'.format(ml.name))

                    ml = create_model(name, params)


def evaluate(name):

    model = tf.keras.models.save_model('{}.h5'.format(name))

    in_seq, out_seq = Data.test_x, Data.test_y

    labels = out_seq.reshape(out_seq.shape[0] * out_seq.shape[1], )

    output = model.model.predict(in_seq)

    output = np.array(output)

    predictions = output.reshape(output.shape[0] * output.shape[1], )

    rmse = model.val_rmse(labels, predictions)
    rse = model.val_rse(labels, predictions)
    corr = model.val_corr(labels, predictions)

    return rmse, rse, corr


def main():

    parser = argparse.ArgumentParser(description='Keras Time series multi-output forecasting')
    parser.add_argument('--n_epoch',  type=list, default=[10, 50, 100, 500, 1000])
    parser.add_argument('--n_kernels_1', type=list, default=[10, 50, 100, 200])
    parser.add_argument('--n_kernels_2', type=list, default=[10, 50, 100])
    parser.add_argument('--window', type=int, default=4)
    parser.add_argument('--horizon', type=int, default=4)
    parser.add_argument('--dropout', type=list, default=[0.0, 0.1, 0.3])
    parser.add_argument('--data_dir', default='../data', type=str)
    params = parser.parse_args()

    global Data

    Data = dataUtil.DataUtils(params.data_dir, params.window, params.horizon)

    validate(params, "LSTM")

    rmse, rse, corr = evaluate("LSTM")

    print("test rmse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(rmse, rse, rse))


if __name__ == "__main__":
    main()
