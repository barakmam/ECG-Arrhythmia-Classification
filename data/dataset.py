import pandas as pd
import numpy as np
import wfdb
import ast



class Dataset:
    def __init__(self):

        self.path = './data/'
        self.sampling_rate = 100

        # Load scp_statements.csv for diagnostic aggregation
        agg_df = pd.read_csv(self.path + 'scp_statements.csv', index_col=0)
        self.agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(self, y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in self.agg_df.index:
                tmp.append(self.agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    def load_raw_data(self, df, sampling_rate, path):
        if sampling_rate == 100:
            data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
        data = np.array([signal for signal, meta in data])
        return data

    def load(self):
        # load and convert annotation data
        Y = pd.read_csv(self.path + 'ptbxl_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        X = self.load_raw_data(Y, self.sampling_rate, self.path)

        # Apply diagnostic superclass
        Y['diagnostic_superclass'] = Y.scp_codes.apply(self.aggregate_diagnostic)

        # Split data into train and test
        test_fold = 10
        # Train
        X_train = X[np.where(Y.strat_fold != test_fold)]
        y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass

        # Test
        X_test = X[np.where(Y.strat_fold == test_fold)]
        y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test
        }