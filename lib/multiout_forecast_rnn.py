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
import numpy as np
import tensorflow as tf
import lib.dataset as data_util
import lib.eval_metrics as eval_metrics
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
        self.model = None
        self.saved_model = None
        self.early_stop = kc.EarlyStopping(monitor='val_loss', patience=20)

    def train(self, epoch, kernel_1, kernel_2, dr):
        pass

    def validate(self):

        in_seq, out_seq = Data.val_x, Data.val_y

        labels = out_seq.reshape(out_seq.shape[0] * out_seq.shape[1], )

        best_val = float("inf")

        for epoch in self.epoch_ls:

            for kernel_1 in self.kernels_1_ls:

                for kernel_2 in self.kernels_2_ls:

                    for dr in self.dropout_ls:

                        self.train(epoch, kernel_1, kernel_2, dr)

                        output = self.model.predict(in_seq)

                        output = np.array(output)

                        predictions = output.reshape(output.shape[0] * output.shape[1], )

                        eval_module = eval_metrics.EvalMetrics(labels, predictions)

                        rmse = eval_module.val_rmse()

                        if rmse < best_val:
                            best_val = rmse
                            self.model.save('{}.h5'.format(self.name))
                            self.saved_model = self.model

    def evaluate(self):

        model = self.saved_model

        in_seq, out_seq = Data.test_x, Data.test_y

        labels = out_seq.reshape(out_seq.shape[0] * out_seq.shape[1], )

        output = model.model.predict(in_seq)

        output = np.array(output)

        predictions = output.reshape(output.shape[0] * output.shape[1], )

        eval_module = eval_metrics.EvalMetrics(labels, predictions)

        rmse = eval_module.val_rmse()
        rse = eval_module.val_rse()
        corr = eval_module.val_corr()

        return rmse, rse, corr


class LSTMModel(DeepModels):

    def __init__(self, params):
        super(LSTMModel, self).__init__(params)
        self.name = "lstm"

    def train(self, epoch, kernels_1, kernels_2, dropout):

        self.model = Sequential()
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

        self.model = Sequential()
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

        self.model = Sequential()
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

        self.model = Sequential()
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

        self.model = Sequential()
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

        self.model = Sequential()
        self.model.add(Bidirectional(GRU(kernels_1, input_shape=(Data.train_x.shape[1], Data.train_x.shape[2]))))
        self.model.add(Dense(kernels_2, activation='relu'))
        self.model.add(Dense(self.horizon, activation='linear'))
        self.model.add(Dropout(rate=dropout))
        self.model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        self.model.fit(Data.train_x, Data.train_y, epochs=epoch, callbacks=[self.early_stop])


def create_models(params):

    model = None
    name = params.name
    if name is "LSTM":
        model = LSTMModel(params)
    elif name is "BiLSTM":
        model = BiLSTMModel(params)
    elif name is "EdLSTM":
        model = EdLSTMModel(params)
    elif name is "BiEdLSTM":
        model = BiEdLSTMModel(params)
    elif name is "CNN":
        model = CNNModel(params)
    elif name is "GRU":
        model = GRUModel(params)
    elif name is "BiGRU":
        model = BiGRUModel(params)

    return model


def main():

    parser = argparse.ArgumentParser(description='Keras Time series multi-output forecasting')
    parser.add_argument('--n_epoch',  type=list, default=[10, 50, 100, 500, 1000])
    parser.add_argument('--n_kernels_1', type=list, default=[10, 50, 100, 200])
    parser.add_argument('--n_kernels_2', type=list, default=[10, 50, 100])
    parser.add_argument('--window', type=int, default=4)
    parser.add_argument('--horizon', type=int, default=4)
    parser.add_argument('--dropout', type=list, default=[0.0, 0.1, 0.3])
    parser.add_argument('--data_dir', default='../data', type=str)
    parser.add_argument('--name', type=str, required=True)
    params = parser.parse_args()

    global Data

    Data = data_util.DataUtils(params)

    Data.train_x = Data.train_x.reshape((Data.train_x.shape[0], Data.train_x.shape[1], 1))

    Data.val_x = Data.val_x.reshape((Data.val_x.shape[0], Data.val_x.shape[1], 1))

    Data.test_x = Data.test_x.reshape((Data.test_x.shape[0], Data.test_x.shape[1], 1))

    model = create_models(params)

    model.validate()

    rmse, rse, corr = model.evaluate()

    print("test rmse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(rmse, rse, rse))


if __name__ == "__main__":
    main()
