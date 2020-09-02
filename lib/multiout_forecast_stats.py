import argparse
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from eval_metrics import EvalMetrics
from dataset import DataOneD
import pickle
global Data


class StatModels:

    def __init__(self, params):

        self.params = params
        self.train = Data.train
        self.validation = Data.validation
        self.test = Data.test
        self.model = None
        self.saved_model = None
        self.start = len(self.train)
        self.end = len(self.train) + len(self.test) - 1
        self.name = None

    def validate(self):
        pass

    def evaluate(self):

        predictions = self.saved_model.predict(start=self.start, end=self.end)
        eval_metrics = EvalMetrics(self.test, predictions)
        rmse = eval_metrics.val_rmse()
        rse = eval_metrics.val_rse()
        corr = eval_metrics.val_corr()

        return rmse, rse, corr

    def save_model(self):
        file_name = self.name + "pkl"
        with open(file_name, 'wb') as file:
            pickle.dump(self.model, file)


class ARIMAModel(StatModels):

    def __init__(self, params):

        super(ARIMAModel, self).__init__(params)
        self.p_ls = params.p_ls
        self.q_ls = params.q_ls
        self.d_ls = params.d_ls
        self.name = "ARIMA"

    def validate(self):
        best_value = float("inf")
        for p in self.p_ls:
            for q in self.q_ls:
                for d in self.d_ls:

                    self.model = ARIMA(self.train, order=(p, q, d))
                    self.model = self.model.fit(disp=0)
                    predictions = self.model.predict(start=self.start, end=self.end)
                    eval_metric = EvalMetrics(self.validation, predictions)
                    rmse = eval_metric.val_rmse()

                    if rmse < best_value:
                        best_value = rmse
                        self.saved_model = self.model


class VARModel(StatModels):

    def __init__(self, params):

        super(VARModel, self).__init__(params)
        self.name = "VAR"

    def validate(self):

        self.model = AutoReg(self.train, lags=len(self.test))
        self.model = self.model.fit()
        self.saved_model = self.model


def create_model(params):

    model = None
    name = params.name

    if name == "ARIMA":
        model = ARIMAModel(params)
    elif name == "VAR":
        model = VARModel(params)

    return model


def main():

    parser = argparse.ArgumentParser(description='statsmodel Time series multi-output forecasting')
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--save', type=str, required=True)
    params = parser.parse_args()

    global Data

    Data = DataOneD(params)
    p_values = [len(Data.test), len(Data.test) * 2]
    q_values = [0, 1, 2]
    d_values = [0, 1, 2]
    parser.add_argument('--p_ls', default=p_values)
    parser.add_argument('--q_ls', default=q_values)
    parser.add_argument('--d_ls', default=d_values)
    parser.add_argument('--name', type=str, required=True)
    params = parser.parse_args()

    model = create_model(params)
    model.validate()

    rmse, rse, corr = model.evaluate()

    with open(params.save, 'a+') as f:

        f.write(f"test rmse {rmse:5.4f} | test rse {rse:5.4f} | test corr {corr:5.4f}")


if __name__ == '__main__':
    main()
