import pickle
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from itertools import compress

def sample_from_unit_cube(feature_dict, N):
    '''Choose without replacement N samples randomly from the dataset that are closest to a randomly generated point on the unit cube'''

    feature_keys = list(feature_dict.keys())
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

    return feature_tensor[sampled], list(compress(feature_keys, sampled))


def sample_furthest_from_centroid(feature_dict, N):
    '''Randomly pick a sample then pick samples that maximise the distance from the current sample's centroid'''
    raise NotImplementedError


def sample_furthest_from_centroid(feature_dict, N):
    '''Randomly pick a sample then pick samples that maximise the distance from the current sample's centroid'''
    raise NotImplementedError


if __name__ == '__main__':
    with open('/home/tom/Pictures/test_pics/mean_color', "rb") as f:
        feature_dict = pickle.load(f)

    sample, _ = sample_from_unit_cube(feature_dict, 1000)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for feature in sample:
        ax.scatter(feature[0], feature[1], feature[2], c=feature, s=2)

    plt.show()
