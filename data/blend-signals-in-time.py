from data.dataset import Dataset
import ecg_plot
import numpy as np


class Blend(Dataset):
    def __init__(self):
        super().__init__()
        data = self.load()
        self.X_train = data["X_train"]
        self.X_train_meta = data["X_train_meta"]
        self.y_train = data["y_train"]
        self.X_test = data["X_test"]
        self.X_test_meta = data["X_test_meta"]
        self.y_test = data["y_test"]
        self.coeff_A = 0.5
        self.coeff_B = 0.5

    def find_pairs(self):
        """
        pairs male and female with the same age(under/above 50)
        :return:
        """

        d = {
            0: {
                "A": [],
                "B": [],
                "Y": []
            },
            1: {
                "A": [],
                "B": [],
                "Y": []
            }
        }

        genders = [0, 1]
        age_th = 50
        mask_below = [True] * (len(self.X_train) // 2) + [False] * (len(self.X_train) // 2)
        mask_above = [False] * (len(self.X_train) // 2) + [True] * (len(self.X_train) // 2)

        for gender in genders:
            #below th
            d[gender]["A"] += [self.X_train[np.isin(self.X_train_meta["sex"], gender) & (self.X_train_meta["age"] < age_th) & mask_below]]
            d[gender]["B"] += [self.X_train[np.isin(self.X_train_meta["sex"], gender) & (self.X_train_meta["age"] < age_th) & mask_above]]

            # above th
            d[gender]["A"] += [self.X_train[np.isin(self.X_train_meta["sex"], gender) & (self.X_train_meta["age"] >= age_th) & mask_below]]
            d[gender]["B"] += [self.X_train[np.isin(self.X_train_meta["sex"], gender) & (self.X_train_meta["age"] >= age_th) & mask_above]]



            # d[gender]["Y"] =

    def plot_ecg(self, ecg):
        ecg_plot.plot(ecg, sample_rate=100, title='ECG 12', columns=1)
        ecg_plot.show()


if __name__ == "__main__":
    b = Blend()
    b.find_pairs()
    # b.plot_ecg(b.X_train[0].T)
