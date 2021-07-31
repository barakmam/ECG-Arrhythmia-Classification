from coding.dataset import Dataset
import ecg_plot
import numpy as np
import itertools
import pickle
import pandas as pd
import scipy.signal as sg
import matplotlib.pyplot as plt
import copy
from google.cloud import storage
import os.path
import ast
import json
import uuid

# Setting credentials using the downloaded JSON file
path = 'model-azimuth-321409-241148a4b144.json'
if not os.path.isfile(path):
    raise ("Please provide the gcs key in the root directory")
client = storage.Client.from_service_account_json(json_credentials_path=path)


class Blend(Dataset):
    def __init__(self):
        super().__init__()
        data = self.load()
        self.DEBUG = False
        self.X_train = data["X_train"]
        self.X_train_meta = data["X_train_meta"]
        self.y_train = data["y_train"]
        self.X_test = data["X_test"]
        self.X_test_meta = data["X_test_meta"]
        self.y_test = data["y_test"]
        self.state = 'STFT'  # expected {STFT,permutation}

        # datasttruct
        self.d = {
            "train": {
                0: {  # male
                    '<': {  # less than 50
                        "A": [],
                        "B": [],
                        "meta_A": [],
                        "meta_B": [],
                        "Y_A": [],
                        "Y_B": []
                    },
                    '>=': {  # over 50
                        "A": [],
                        "B": [],
                        "meta_A": [],
                        "meta_B": [],
                        "Y_A": [],
                        "Y_B": []
                    }

                },
                1: {  # female
                    '<': {  # less than 50
                        "A": [],
                        "B": [],
                        "meta_A": [],
                        "meta_B": [],
                        "Y_A": [],
                        "Y_B": []
                    },
                    '>=': {  # over 50
                        "A": [],
                        "B": [],
                        "meta_A": [],
                        "meta_B": [],
                        "Y_A": [],
                        "Y_B": []
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
                        "Y_A": [],
                        "Y_B": []
                    },
                    '>=': {  # over 50
                        "A": [],
                        "B": [],
                        "meta_A": [],
                        "meta_B": [],
                        "Y_A": [],
                        "Y_B": []
                    }

                },
                1: {  # female
                    '<': {  # less than 50
                        "A": [],
                        "B": [],
                        "meta_A": [],
                        "meta_B": [],
                        "Y_A": [],
                        "Y_B": []
                    },
                    '>=': {  # over 50
                        "A": [],
                        "B": [],
                        "meta_A": [],
                        "meta_B": [],
                        "Y_A": [],
                        "Y_B": []
                    }
                }

            }
        }

        # Blend
        self.coeff_A = 0.5  # <-- how much does A effect the blending
        self.coeff_B = 0.5  # <-- how much does B effect the blending
        self.dataset_types = ["train", "test"]
        self.genders = [0, 1]
        self.ops = ["<", ">="]
        self.age_th = 50

        # STFT
        self.STFT_show = False
        self.STFT_gender = 0
        self.STFT_op = "<"
        self.hop = 10000
        self.win = 1024
        self.F = 512
        self.resample_rate = 3600
        self.sample_rate = 100
        self.chunck_size = 2

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

                    d[dataset_type][gender][op]["Y_A"], d[dataset_type][gender][op]["Y_B"] = a, b
                    d[dataset_type][gender][op]["A"], d[dataset_type][gender][op]["B"] = self.trancate(
                        d[dataset_type][gender][op]["A"], d[dataset_type][gender][op]["B"])
                    d[dataset_type][gender][op]["meta_A"], d[dataset_type][gender][op]["meta_B"] = self.trancate(
                        d[dataset_type][gender][op]["meta_A"],
                        d[dataset_type][gender][op]["meta_B"])

        return d

    def STFT(self, signal, win, hopSize, F, Fs):
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

            if self.DEBUG and startIdx + len(win) > len(signal):
                print("iteration:True".format())

        stft = np.stack(stft).T
        return stft

    def gender_str(self, gender):
        return 'male' if gender == 0 else 'female'

    def gcs_bucket(self, d):
        """
        According to the state, save a string that represent the data set.
        It is a string that can be later converterd to a dict. It contains the following groups {male,female},{<,>=}
        :param d: a dict containing the {male,female},{<,>=},{A,B,Y}
        :return: None
        """

        # Creating bucket object
        bucket = client.get_bucket('ecg-arrhythmia-classification')
        pkl_dict = copy.deepcopy(self.d)

        print("start gcs_bucket!")

        for dataset_type in self.dataset_types:
            for gender in self.genders:
                for op in self.ops:
                    if self.state == "permutation":
                        for r, idx in zip(
                                itertools.product(d[dataset_type][gender][op]["A"],
                                                  d[dataset_type][gender][op]["B"]),
                                [a for a in itertools.product(*[range(len(x)) for x in
                                                                [d[dataset_type][gender][op]["A"],
                                                                 d[dataset_type][gender][op]["B"]]])]):

                            for idx_single, single in enumerate(["A", "B"]):
                                pkl_dict[dataset_type][gender][op][single].append(r[idx_single])
                                pkl_dict[dataset_type][gender][op][f"meta_{single}"].append(
                                    d[gender][op][f"meta_{single}"][idx[idx_single]])
                                pkl_dict[dataset_type][gender][op][f"Y_{single}"].append(
                                    d[gender][op][f"Y_{single}"][idx[idx_single]])


                    elif self.state == "STFT":
                        length = len(d[dataset_type][gender][op]["A"])
                        print(
                            "FROM dataset_type:{}, gender:{}, op:{}".format(dataset_type, self.gender_str(gender), op))
                        for index in range(length):
                            print("Now processing STFT img:{}/{} ".format(index + 1, length))

                            for single in ["A", "B"]:

                                ecg = d[dataset_type][gender][op][single][index].T[0]

                                f, t, Zxx = sg.stft(ecg, fs=100, nperseg=512, noverlap=512 - 1)
                                # f, t, Zxx = sg.stft(rec["'MLII'"][:1024], fs=360, nperseg=512, noverlap=0)
                                # plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')

                                X_stft = self.STFT(sg.resample(ecg, 360000,
                                                               np.arange(0, 1000) / 100)[0],
                                                   self.win, self.hop, self.F,
                                                   self.sample_rate * self.resample_rate / 1000)

                                tau = np.arange(X_stft.shape[1]) * self.hop / self.sample_rate
                                freqs = np.fft.fftshift(np.fft.fftfreq(self.F, 1 / self.sample_rate))
                                im = plt.pcolormesh(tau, freqs, np.fft.fftshift(np.abs(X_stft), axes=0))

                                # Y
                                y = d[dataset_type][gender][op][f"Y_{single}"].iloc[index][0]
                                pkl_dict[dataset_type][gender][op][f"Y_{single}"].append(y)

                                # create the dataset: data+metadata
                                file_uuided = str(uuid.uuid4())


                                plt.ylabel('f [Hz]', fontsize=16)
                                plt.xlabel('$\\tau$ [sec]', fontsize=16)
                                plt.title(
                                    'win:{} , hopSize:{} , F:{}, y:{}'.format(self.win, self.hop, self.F, y),
                                    fontsize=16)
                                plt.colorbar(im)
                                plt.suptitle('| STFT(f, $\\tau$) |', fontsize=16)
                                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                                with open('myplot.pkl', 'wb') as fid:
                                    pickle.dump(im, fid)

                                blob = bucket.blob('{}/{}'.format(self.state, file_uuided))
                                with open("./myplot.pkl", 'rb') as f:
                                    blob.upload_from_file(f)

                                pkl_dict[dataset_type][gender][op][single].append("STFT/"+file_uuided)
                                pkl_dict[dataset_type][gender][op][f"meta_{single}"].append(
                                    d[dataset_type][gender][op][f"meta_{single}"][index])



                                if self.STFT_show:
                                    plt.show()

                    else:
                        raise NotImplementedError


        object_name_in_gcs_bucket = bucket.blob('state:{}'.format(self.state))
        object_name_in_gcs_bucket.upload_from_string(str(pkl_dict))

    def load_dataset(self):
        """
        https://stackoverflow.com/questions/7290370/store-and-reload-matplotlib-pyplot-object
        """
        bucket = client.get_bucket('ecg-arrhythmia-classification')
        blob = bucket.blob('state:{}'.format(self.state))
        d = ast.literal_eval(blob.download_as_string().decode('utf-8'))

        blob=bucket.blob("{}/{}".format(self.state,d['train'][0]['<']['A'][0]))
        pickle.loads(blob.download_as_bytes())
        plt.show()


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
        Y_A = gender[op]["Y_A"][index]
        Y_B = gender[op]["Y_B"][index]

        ecg_plot.plot(A, sample_rate=100, title="{}-{}-{}-{}".format(meta_A, gender_str, op_str, Y_A), columns=1)
        ecg_plot.plot(B, sample_rate=100, title="{}-{}-{}-{}".format(meta_B, gender_str, op_str, Y_B), columns=1)

        # <-- this is where the blending should happen

        # C=self.blend_in_time(A,B)

        ecg_plot.show()


if __name__ == "__main__":
    b = Blend()
    pairs = b.find_pairs()
    # b.gcs_bucket(pairs)
    b.load_dataset()
    # b.blend_and_plot_ecg(pairs, 0)
