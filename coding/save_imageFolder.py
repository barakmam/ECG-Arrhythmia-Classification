import os

import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import scipy.signal as sg
import seaborn as sb
import pandas as pd
from PIL import Image
import cv2
from tqdm import tqdm

base_path = '/inputs/TAU/SP/data/stft_norm/'

super_classes = ["CD", "HYP", "MI", "NORM", "STTC"]
files_names = os.listdir(base_path + 'images')
print('len source: ', len(files_names))

for ii, file_name in enumerate(files_names):
    label = super_classes[int(file_name.split('label_')[1][0])]
    os.rename(base_path + 'images/' + file_name, base_path + label + '/' + file_name)

print('len source: ', len(os.listdir(base_path + 'images')))
for label in super_classes:
    label_dir = os.listdir(base_path + label)
    print('len ' + label + ': ', len(label_dir))

