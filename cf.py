import pandas as pd
import numpy as np
from PIL import Image
import os
from numpy import asarray
from tqdm import tqdm

os.chdir('/home/ubuntu/lab3')

#
# labels = pd.read_csv('test_label.csv', header=None)[0].tolist()
# uniques, freq = np.unique(labels, return_counts=True)
# print('All testing data: ', sum(freq))
# print('Class 0 ratio: ', freq[0]/sum(freq))

#
names = pd.read_csv('train_img.csv', header=None)[0].tolist()

all_means_c1 = []
all_means_c2 = []
all_means_c3 = []
for name in tqdm(names):
    img = Image.open(os.path.join('/home/ubuntu/lab3/data', name + '.jpeg'))
    means = asarray(img).mean(axis=(0, 1), dtype='float64')
    all_means_c1.append(means[0])
    all_means_c2.append(means[1])
    all_means_c3.append(means[2])
