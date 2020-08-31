from math import sqrt
import numpy as np
from sklearn.metrics import mean_squared_error


class EvalMetrics:

    def __init__(self, labels, predictions):

        self.labels = labels
        self.predictions = predictions

    def val_rse(self):

        test_rse = 0.0
        actual_var = 0.0
        all_outputs = np.array(self.predictions)
        all_labels = np.array(self.labels)
        mean_labels = all_labels.mean(axis=0)
        test_rse += np.sum((all_labels - all_outputs) ** 2)
        actual_var += np.sum((all_labels - mean_labels) ** 2)
        rse = test_rse / actual_var
        rse = float("{:.4f}".format(rse))
        return rse

    def val_rmse(self):

        mse = mean_squared_error(self.labels, self.predictions)
        rmse = sqrt(mse)
        rmse = float("{:.4f}".format(rmse))
        return rmse

    def val_corr(self):

        all_outputs = np.array(self.predictions)
        all_labels = np.array(self.labels)
        sigma_outputs = all_outputs.std(axis=0)
        sigma_labels = all_labels.std(axis=0)
        mean_outputs = all_outputs.mean(axis=0)
        mean_labels = all_labels.mean(axis=0)
        idx = sigma_labels != 0
        test_corr = (
                            (all_outputs - mean_outputs) * (all_labels - mean_labels)
                    ).mean(axis=0) / (sigma_outputs * sigma_labels)
        corr = test_corr[idx].mean()
        corr = float("{:.4f}".format(corr))
        return corr
