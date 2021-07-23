from data.dataset import Dataset
import ecg_plot
import numpy as np
import itertools
import pickle
import pandas as pd

##
class Blend(Dataset):
    def __init__(self):
        super().__init__()
        data = self.load()
        self.pre_load_pkl_data = False
        self.pkl_file_relative_path = "./data/data.pickle"
        self.X_train = data["X_train"]
        self.X_train_meta = data["X_train_meta"]
        self.y_train = data["y_train"]
        self.X_test = data["X_test"]
        self.X_test_meta = data["X_test_meta"]
        self.y_test = data["y_test"]

        self.coeff_A = 0.5 # <-- how much does A effect the blending
        self.coeff_B = 0.5 # <-- how much does B effect the blending
        self.genders = [0, 1]
        self.ops = ["<", ">="]
        self.age_th = 50

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

        mask_below = np.array(list(range(len(self.X_train)))) < len(self.X_train) // 2
        mask_above = list(reversed(mask_below))

        for gender in self.genders:
            for op in self.ops:
                mask_A = np.isin(self.X_train_meta["sex"], gender) & self.custom_operator(
                    self.X_train_meta["age"], self.age_th, op) & mask_below

                mask_B = np.isin(self.X_train_meta["sex"], gender) & self.custom_operator(
                    self.X_train_meta["age"], self.age_th, op) & mask_above

                # features
                d[gender][op]["A"] = self.X_train[mask_A]
                d[gender][op]["meta_A"] = pd.DataFrame(self.X_train_meta)[mask_A].to_dict('records')

                d[gender][op]["B"] = self.X_train[mask_B]
                d[gender][op]["meta_B"] = pd.DataFrame(self.X_train_meta)[mask_B].to_dict('records')

                # labels
                a = self.y_train[
                    np.isin(self.X_train_meta["sex"], gender) & self.custom_operator(self.X_train_meta["age"],
                                                                                     self.age_th,
                                                                                     op) & mask_below]
                b = self.y_train[
                    np.isin(self.X_train_meta["sex"], gender) & self.custom_operator(self.X_train_meta["age"],
                                                                                     self.age_th,
                                                                                     op) & mask_above]

                # trancate
                a, b = self.trancate(a, b)

                d[gender][op]["Y"] = [np.vstack((a, b))]
                d[gender][op]["A"], d[gender][op]["B"] = self.trancate(d[gender][op]["A"], d[gender][op]["B"])
                d[gender][op]["meta_A"], d[gender][op]["meta_B"] = self.trancate(d[gender][op]["meta_A"],
                                                                                 d[gender][op]["meta_B"])

        return d

    def permutate_and_pkl(self, d):
        """
        Create a pkl file which contains the permutations of the following groups {male,female},{<,>=}
        :param d: a dict containing the {male,female},{<,>=},{A,B,Y}
        :return: a dict containing the {male,female},{<,>=},{A,B,Y} permutated
        """

        pkl_dict = {
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
        if not self.pre_load_pkl_data:
            for gender in self.genders:
                for op in self.ops:
                    for r, idx in zip(itertools.product(d[gender][op]["A"], d[gender][op]["B"], d[gender][op]["Y"]),
                                      [a for a in itertools.product(*[range(len(x)) for x in
                                                                      [d[gender][op]["A"], d[gender][op]["B"],
                                                                       d[gender][op]["Y"]]])]):
                        pkl_dict[gender][op]["A"].append(r[0])
                        pkl_dict[gender][op]["meta_A"].append(d[gender][op]["meta_A"][idx[0]])
                        pkl_dict[gender][op]["B"].append(r[1])
                        pkl_dict[gender][op]["meta_B"].append(d[gender][op]["meta_B"][idx[1]])
                        pkl_dict[gender][op]["Y"].append(r[2])

            with open(self.pkl_file_relative_path, 'wb') as handle:
                pickle.dump(pkl_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(self.pkl_file_relative_path, 'rb') as handle:
                pkl_dict = pickle.load(handle)

        return pkl_dict


    def blend_in_time(self,A,B):
        """
        Blend two matricies in time
        :param A: 12x1000 ndarray
        :param B: 12x1000 ndarray
        :return: C 12x1000 ndarray blended with self.coeff_A and self.coeff_B ratios
        """

        a=A[0]
        b=B[0]

        raise NotImplemented
    def blend_and_plot_ecg(self, pairs,index):
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

        #<-- this is where the blending should happen
        #C=self.blend_in_time(A,B)

        ecg_plot.show()


if __name__ == "__main__":
    b = Blend()
    pairs = b.find_pairs()
    pairs = b.permutate_and_pkl(pairs)
    b.blend_and_plot_ecg(pairs,0)
