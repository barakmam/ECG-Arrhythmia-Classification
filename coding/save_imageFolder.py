import os

import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import scipy.signal as sg
# import seaborn as sb
import pandas as pd
from PIL import Image
# import cv2
from tqdm import tqdm

# base_path = '/inputs/TAU/SP/data/stft_norm/'
#
# super_classes = ["CD", "HYP", "MI", "NORM", "STTC"]
# files_names = os.listdir(base_path + 'images')
# print('len source: ', len(files_names))
#
# for ii, file_name in enumerate(files_names):
#     label = super_classes[int(file_name.split('label_')[1][0])]
#     os.rename(base_path + 'images/' + file_name, base_path + label + '/' + file_name)
#
# print('len source: ', len(os.listdir(base_path + 'images')))
# for label in super_classes:
#     label_dir = os.listdir(base_path + label)
#     print('len ' + label + ': ', len(label_dir))
#


def loader(path):
    s = " ".join(open(path, "r").read().split())
    wavelet = ast.literal_eval(s.replace('\n', '').replace(' ', ','))
    return wavelet

base_path = '/Users/barakm/Desktop/TAU/signal_processing/ECG-Arrhythmia-Classification/data/MorlWavelet/full_with_single'
classes_names = os.listdir(base_path)[1:]
dest_path = '/Users/barakm/Desktop/TAU/signal_processing/ECG-Arrhythmia-Classification/data/MorlWavelet/full_with_single_uint8'

num = 3  # remains 3
os.makedirs(dest_path, exist_ok=True)
for cls in [classes_names[num]]:
    print('class: ', cls)
    files = os.listdir(os.path.join(base_path, cls))
    os.makedirs(os.path.join(dest_path, cls), exist_ok=True)
    iFile = 0
    for file in tqdm(files):
        data = loader(os.path.join(base_path, cls, file))
        data_uint8 = (np.array(data)*255).astype(np.uint8)
        np.save(os.path.join(dest_path, cls, file), data_uint8, allow_pickle=True)

        iFile += 1
    print('Finished class: ', cls)
    print('class num: ', num)
# a = 1
# np.save('/Users/barakm/Desktop/TAU/signal_processing/ECG-Arrhythmia-Classification/data/MorlWavelet/full_with_single/test_uint8', c, allow_pickle=True)
# b = np.unique(a)
