from data.dataset import Dataset
import ecg_plot
import numpy as np
import ast


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

    def custom_operator(self, a, b, op):
        if op == "<":
            return a < b
        elif op == ">=":
            return a >= b

    def trancate(self, a, b):
        if len(a) != len(b):
            if len(a) > len(b):
                a = a[:-1]
            else:
                b = b[:-1]

        return a, b

    def find_pairs(self):
        """
        pairs male with the same age(under/above 50)
        and female with the same age(under/above 50)
        :return:
        """

        d = {
            0: {  # male
                '<': {  # less than 50
                    "A": [],
                    "B": [],
                    "Y": []
                },
                '>=': {  # over 50
                    "A": [],
                    "B": [],
                    "Y": []
                }

            },
            1: {  # female
                '<': {  # less than 50
                    "A": [],
                    "B": [],
                    "Y": []
                },
                '>=': {  # over 50
                    "A": [],
                    "B": [],
                    "Y": []
                }
            }
        }

        genders = [0, 1]
        ops = ["<", ">="]
        age_th = 50
        mask_below = np.array(list(range(len(self.X_train)))) < len(self.X_train) // 2
        mask_above = list(reversed(mask_below))

        for gender in genders:
            for op in ops:
                d[gender][op]["A"] = self.X_train[np.isin(self.X_train_meta["sex"], gender) & self.custom_operator(
                    self.X_train_meta["age"], age_th, op) & mask_below]
                d[gender][op]["B"] = self.X_train[np.isin(self.X_train_meta["sex"], gender) & self.custom_operator(
                    self.X_train_meta["age"], age_th, op) & mask_above]

                d[gender][op]["A"], d[gender][op]["B"] = self.trancate(d[gender][op]["A"], d[gender][op]["B"])

                a = self.y_train[
                    np.isin(self.X_train_meta["sex"], gender) & self.custom_operator(self.X_train_meta["age"], age_th,
                                                                                     op) & mask_below]
                b = self.y_train[
                    np.isin(self.X_train_meta["sex"], gender) & self.custom_operator(self.X_train_meta["age"], age_th,
                                                                                     op) & mask_above]

                a, b = self.trancate(a, b)

                d[gender][op]["Y"] = [np.vstack((a, b))]

        return d

    def plot_ecg(self, pairs):

        gender = 0  # 0 male ; 1 female
        gender_str="male" if not gender else "female"
        op = '<'
        op_str="under 50" if op=='<' else "above 50"
        index = 0
        gender = pairs[gender]

        A = gender[op]["A"][index].T
        B = gender[op]["B"][index].T
        Y_A = gender[op]["Y"][index][0][0][0]
        Y_B = gender[op]["Y"][index][1][0][0]

        ecg_plot.plot(A, sample_rate=100, title="A-{}-{}-{}".format(gender_str,op_str, Y_A), columns=1)
        ecg_plot.plot(B, sample_rate=100, title="B-{}-{}-{}".format(gender_str,op_str, Y_B), columns=1)
        ecg_plot.show()


if __name__ == "__main__":
    b = Blend()
    pairs = b.find_pairs()
    b.plot_ecg(pairs)
