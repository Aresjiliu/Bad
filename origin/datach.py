import os
import numpy as np
import torch
import csv
import cv2
import random
import scipy.io as scio
import h5py
import mat73


def load_mat(mat_path):
    try:
        data = scio.loadmat(mat_path)
        data = data[list(data.keys())[-1]]
    except Exception as e:
        data = mat73.loadmat(mat_path)
        data = data[list(data.keys())[0]]
    return data


label_path = "/data1/yco/TGRS/data/Huston2013/hsi/hsi_Y_test.mat"
arr = np.array(load_mat(label_path)).flatten()
counts = np.bincount(arr)

# 打印每个数字的数量
for i in range(len(counts)):
    print(f"数字 {i} 的数量是: {counts[i]}")
