from data.dataset import Dataset
import ecg_plot
import numpy as np
import itertools
import pickle
import pandas as pd
import scipy.signal as sg
import matplotlib.pyplot as plt
import copy


##
class Blend(Dataset):
    def __init__(self):
        super().__init__()
        data = self.load()
        self.pre_load_pkl_data = False
        self.X_train = data["X_train"]
        self.X_train_meta = data["X_train_meta"]
        self.y_train = data["y_train"]
        self.X_test = data["X_test"]
        self.X_test_meta = data["X_test_meta"]
        self.y_test = data["y_test"]

        # datasttruct
        self.d = {
            "train": {
                0: {  # male
                    '<': {  # less than 50
                        "A": [],
                        "B": [],
                        "meta_A": [],
                        "meta_B": [],
                        "Y": []
                    },
                    '>=': {  # over 50
                        "A": [],
                        "B": [],
                        "meta_A": [],
                        "meta_B": [],
                        "Y": []
                    }

                },
                1: {  # female
                    '<': {  # less than 50
                        "A": [],
                        "B": [],
                        "meta_A": [],
                        "meta_B": [],
                        "Y": []
                    },
                    '>=': {  # over 50
                        "A": [],
                        "B": [],
                        "meta_A": [],
                        "meta_B": [],
                        "Y": []
                    }
                }
            },
            "test": {
                0: {  # male
                    '<': {  # less than 50
                        "A": [],
                        "B": [],
                        "meta_A": [],
                        "meta_B": [],
                        "Y": []
                    },
                    '>=': {  # over 50
                        "A": [],
                        "B": [],
                        "meta_A": [],
                        "meta_B": [],
                        "Y": []
                    }

                },
                1: {  # female
                    '<': {  # less than 50
                        "A": [],
                        "B": [],
                        "meta_A": [],
                        "meta_B": [],
                        "Y": []
                    },
                    '>=': {  # over 50
                        "A": [],
                        "B": [],
                        "meta_A": [],
                        "meta_B": [],
                        "Y": []
                    }
                }

            }
        }

        self.coeff_A = 0.5  # <-- how much does A effect the blending
        self.coeff_B = 0.5  # <-- how much does B effect the blending
        self.dataset_types = ["train", "test"]
        self.genders = [0, 1]
        self.ops = ["<", ">="]
        self.age_th = 50

        # STFT
        self.STFT_show=True
        self.STFT_gender = 0
        self.STFT_op = "<"
        self.hop = 1
        self.win = 1024
        self.F = 512
        self.resample_rate = 360000
        self.sample_rate = 100

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

    def feature_label_selection(self, d, gender, op, dataset_type):
        """
        extract the relevant features and lables both for train and predict
        """

        X = self.X_train if dataset_type == "train" else self.X_test
        X_meta = self.X_train_meta if dataset_type == "train" else self.X_test_meta
        Y = self.y_train if dataset_type == "train" else self.y_test

        mask_below = np.array(list(range(len(X)))) < len(X) // 2
        mask_above = list(reversed(mask_below))

        mask_A = np.isin(X_meta["sex"], gender) & self.custom_operator(
            X_meta["age"], self.age_th, op) & mask_below

        mask_B = np.isin(X_meta["sex"], gender) & self.custom_operator(
            X_meta["age"], self.age_th, op) & mask_above

        # features
        d[dataset_type][gender][op]["A"] = X[mask_A]
        d[dataset_type][gender][op]["meta_A"] = pd.DataFrame(X_meta)[mask_A].to_dict('records')

        d[dataset_type][gender][op]["B"] = X[mask_B]
        d[dataset_type][gender][op]["meta_B"] = pd.DataFrame(X_meta)[mask_B].to_dict('records')

        # labels
        a = Y[
            np.isin(X_meta["sex"], gender) & self.custom_operator(X_meta["age"],
                                                                  self.age_th,
                                                                  op) & mask_below]
        b = Y[
            np.isin(X_meta["sex"], gender) & self.custom_operator(X_meta["age"],
                                                                  self.age_th,
                                                                  op) & mask_above]

        return a, b

    def find_pairs(self):
        """
        pairs male with the same age(under/above 50)
        and female with the same age(under/above 50)
        :return:
        """

        d = copy.deepcopy(self.d)

        for dataset_type in self.dataset_types:
            for gender in self.genders:
                for op in self.ops:
                    a, b = self.feature_label_selection(d=d, gender=gender, op=op, dataset_type=dataset_type)

                    # trancate
                    a, b = self.trancate(a, b)

                    d[dataset_type][gender][op]["Y"] = [np.vstack((a, b))]
                    d[dataset_type][gender][op]["A"], d[dataset_type][gender][op]["B"] = self.trancate(
                        d[dataset_type][gender][op]["A"], d[dataset_type][gender][op]["B"])
                    d[dataset_type][gender][op]["meta_A"], d[dataset_type][gender][op]["meta_B"] = self.trancate(
                        d[dataset_type][gender][op]["meta_A"],
                        d[dataset_type][gender][op]["meta_B"])

        return d

    def STFT(signal, win, hopSize, F, Fs):
        if not hasattr(win, "__len__"):
            win = np.hamming(win)
        if not hasattr(F, "__len__"):
            F = 2 * np.pi * np.arange(F) / F

        t = np.arange(len(signal))

        stft = []
        startIdx = 0
        while startIdx + len(win) <= len(signal):
            e = np.exp(
                -1j * t[startIdx:(startIdx + len(win))].reshape(1, -1) * F.reshape(-1, 1))
            currDFT = np.sum(signal[startIdx:(startIdx + len(win))] * win * e, 1)
            stft.append(np.abs(currDFT).astype(np.complex64))
            startIdx += hopSize

        stft = np.stack(stft).T
        return stft

    def pkl(self, d, state="STFT"):
        """
        Create a pkl file which contains the permutations of the following groups {male,female},{<,>=}
        :param d: a dict containing the {male,female},{<,>=},{A,B,Y}
        :return: a dict containing the {male,female},{<,>=},{A,B,Y} permutated
        """

        pkl_dict = copy.deepcopy(self.d)
        if not self.pre_load_pkl_data:
            for dataset_type in self.dataset_types:
                for gender in self.genders:
                    for op in self.ops:
                        if state == "permutate":
                            for r, idx in zip(
                                    itertools.product(d[dataset_type][gender][op]["A"],
                                                      d[dataset_type][gender][op]["B"],
                                                      d[dataset_type][gender][op]["Y"]),
                                    [a for a in itertools.product(*[range(len(x)) for x in
                                                                    [d[dataset_type][gender][op]["A"],
                                                                     d[dataset_type][gender][op]["B"],
                                                                     d[dataset_type][gender][op]["Y"]]])]):
                                pkl_dict[gender][op]["A"].append(r[0])
                                pkl_dict[gender][op]["meta_A"].append(d[gender][op]["meta_A"][idx[0]])
                                pkl_dict[gender][op]["B"].append(r[1])
                                pkl_dict[gender][op]["meta_B"].append(d[gender][op]["meta_B"][idx[1]])
                                pkl_dict[gender][op]["Y"].append(r[2])
                        elif state == "STFT":

                            for index in range(len(d[dataset_type][gender][op]["A"])):
                                for single in ["A", "B"]:

                                    ecg = d[dataset_type][gender][op][single][index].T

                                    f, t, Zxx = sg.stft(ecg, fs=100, nperseg=512,
                                                        noverlap=512 - 1)
                                    # f, t, Zxx = sg.stft(rec["'MLII'"][:1024], fs=360, nperseg=512, noverlap=0)
                                    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')

                                    X_stft = self.STFT(sg.resample(ecg, 360000,
                                                                   np.arange(0, 1000) / 100)[0],
                                                       self.win, self.hop, self.F,
                                                       self.sample_rate * self.resample_rate / 1000)

                                    tau = np.arange(X_stft.shape[1]) * self.hop / self.sample_rate
                                    freqs = np.fft.fftshift(np.fft.fftfreq(self.F, 1 / self.sample_rate))
                                    im = plt.pcolormesh(tau, freqs, np.fft.fftshift(np.abs(X_stft), axes=0))

                                    pkl_dict[dataset_type][gender][op][single].append(im)

                                    if self.STFT_show:
                                        plt.ylabel('f [Hz]', fontsize=16)
                                        plt.xlabel('$\\tau$ [sec]', fontsize=16)
                                        plt.title('win: ' + str(self.win) + '   hopSize: ' + str(self.hop) + '   F: ' + str(
                                            self.F),
                                                  fontsize=16)
                                        plt.colorbar(im)
                                        plt.suptitle('| STFT(f, $\\tau$) |', fontsize=16)
                                        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                                        plt.show()

                        else:
                            raise NotImplementedError

            with open(f"./data/pkl/{state}_data.pickle", 'wb') as handle:
                pickle.dump(pkl_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(f"./data/pkl/{state}_data.pickle", 'rb') as handle:
                pkl_dict = pickle.load(handle)

        return pkl_dict

    def blend_in_time(self, A, B):
        """
        Blend two matricies in time
        :param A: 12x1000 ndarray
        :param B: 12x1000 ndarray
        :return: C 12x1000 ndarray blended with self.coeff_A and self.coeff_B ratios
        """

        a = A[0]
        b = B[0]

        raise NotImplemented

    def blend_and_plot_ecg(self, pairs, index):
        """
        Display the ecg of a selected index
        :param pairs: a dict containing the {male,female},{<,>=},{A,B,Y} permutated
        :param index: the index of the pair for which we will preform the blending
        :return: None
        """

        gender = 0  # 0 male ; 1 female
        gender_str = "male" if not gender else "female"
        op = '<'
        op_str = "under 50" if op == '<' else "above 50"
        gender = pairs[gender]

        A = gender[op]["A"][index].T
        meta_A = gender[op]["meta_A"][index]
        B = gender[op]["B"][index].T
        meta_B = gender[op]["meta_B"][index]
        Y_A = gender[op]["Y"][index][0][0][0]
        Y_B = gender[op]["Y"][index][1][0][0]

        ecg_plot.plot(A, sample_rate=100, title="{}-{}-{}-{}".format(meta_A, gender_str, op_str, Y_A), columns=1)
        ecg_plot.plot(B, sample_rate=100, title="{}-{}-{}-{}".format(meta_B, gender_str, op_str, Y_B), columns=1)

        # <-- this is where the blending should happen

        # C=self.blend_in_time(A,B)

        ecg_plot.show()


if __name__ == "__main__":
    b = Blend()
    pairs = b.find_pairs()
    pairs = b.pkl(pairs, state="STFT")
    b.blend_and_plot_ecg(pairs, 0)
