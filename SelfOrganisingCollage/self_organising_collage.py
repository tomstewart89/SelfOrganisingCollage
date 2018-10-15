import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import random
from tqdm import tqdm
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from photo_library import PhotoLibrary
from SOM import SelfOrganisingMap
from feature_extraction import FeatureExtractor
from draw_samples import sample_from_unit_cube
from patch_worker import Patchworker
from utils import ravel_index, simplify_shape


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make a lovely colourful collage from your photo library')
    parser.add_argument('--directory', default='/home/tom/Pictures/test_pics', type=str)
    parser.add_argument('--width', default=50, type=int)
    parser.add_argument('--height', default=35, type=int)
    parser.add_argument('--feature', default='mean_color', type=str)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--reuse_penalty', default=100., type=float)
    parser.add_argument('--sample_size', default=1000, type=int)
    parser.add_argument('--border', default=20, type=int)
    parser.add_argument('--magnify', default=200, type=int)
    args = parser.parse_args()

    library = PhotoLibrary(args.directory)
    extractor = FeatureExtractor.factory(args.feature)
    som = SelfOrganisingMap(shape=[args.height, args.width, extractor.feature_dim], sigma=5., eta=1.)
    patch = Patchworker([args.height, args.width])

    try:
        with open(os.path.join(args.directory, args.feature), "rb") as f:
            feature_dict = pickle.load(f)

    except FileNotFoundError:
        feature_dict = extractor.process_library(library)
        with open(os.path.join(args.directory, args.feature), "wb") as f:
            pickle.dump(feature_dict, f)

    # draw a subset of just the most interesting the entire photo library
    sample, keys = sample_from_unit_cube(feature_dict, args.sample_size)

    for _ in range(args.epochs):
        for feature in sample[torch.randperm(args.sample_size),:]:
            som.update(feature.cuda())

    # now find the best vibing photo and stick it at its BMU
    times_used = torch.zeros(len(sample)).unsqueeze(1).cuda()
    layout = []

    # I can omit the occupied parts of the grid to speed this up
    while not patch.full():
        dist = (sample.cuda().unsqueeze(1) - som.grid.reshape(-1,3).unsqueeze(0)).norm(dim=2)
        dist += (times_used * args.reuse_penalty) # penalise the samples that have already been placed on the grid
        dist[:, patch.occupied.reshape(-1)] = dist.max() # omit any neurons that already have a photo covering them

        # find the neuron and sample that match most closely
        winning_sample, winning_neuron = ravel_index(dist.argmin(), (som.grid.shape[0] * som.grid.shape[1]))
        target_coord = torch.Tensor(ravel_index(winning_neuron, som.grid.shape[1]))
        target_shape = simplify_shape(library[keys[winning_sample]].size)

        # add the photo to the patch work
        coord, shape = patch.add_patch(target_coord, target_shape)

        layout.append([keys[winning_sample], coord, shape])

        times_used[winning_sample] += 1
        print('tick')


    canvas = PIL.Image.new("RGB", np.multiply(som.grid.shape[:2], args.magnify) + [args.border, args.border], "white")

    for key, coord, shape in layout:
        img = library[key]

        size_tup = shape * args.magnify - args.border
        coord_tup = coord * args.magnify + args.border

        # crop the image so that it matches its shape
        centroid = np.divide(img.size, 2)
        scale = min(img.size[0] / shape[0], img.size[1] / shape[1])
        dims = np.multiply(shape[i], scale)
        box = (centroid[0] - dims[0] / 2, centroid[1] - dims[1] / 2, centroid[0] + dims[0] / 2, centroid[1] + dims[1] / 2)

        canvas.paste(img.crop(box).resize(size_tup, PIL.Image.NEAREST), coord_tup)

    plt.imshow(canvas)
    plt.show()

    canvas.save('invite.png')
