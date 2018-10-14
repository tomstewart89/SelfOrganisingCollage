import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import random
from tqdm import tqdm
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def sample_from_unit_cube(feature_dict, N):
    '''Choose without replacement N samples randomly from the dataset that are closest to a randomly generated point on the unit cube'''
    feature_tensor = torch.stack(list(feature_dict.values()))
    sampled = torch.zeros(len(feature_tensor), dtype=torch.uint8)

    for _ in range(N):
        # generate a point on the unit cube
        point = torch.rand(feature_tensor.shape[1])
        point[0] = point[0].round()
        point = point[torch.randperm(feature_tensor.shape[1])]

        dist_from_point = (feature_tensor - point).norm(dim=1)
        dist_from_point[sampled] = dist_from_point.max()

        sampled[dist_from_point.argmin()] = 1

    return feature_tensor[sampled]

def sample_furthest_from_centroid(feature_dict, N):
    '''Randomly pick a sample then pick samples that maximise the distance from the current sample's centroid'''
    pass

