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
from draw_samples import sample_from_unit_cube, sample_furthest_from_centroid


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make a lovely colourful collage from your photo library')
    parser.add_argument('--directory', default='/home/tom/Pictures/test_pics', type=str)
    parser.add_argument('--width', default=50, type=int)
    parser.add_argument('--height', default=35, type=int)
    parser.add_argument('--feature', default='mean_color', type=str)
    args = parser.parse_args()

    library = PhotoLibrary(args.directory)
    extractor = FeatureExtractor.factory(args.feature)
    som = SelfOrganisingMap(shape=[args.height, args.width, extractor.feature_dim], sigma=5., eta=1.)

    try:
        with open(os.path.join(args.directory, args.feature), "rb") as f:
            feature_dict = pickle.load(f)

    except FileNotFoundError:
        feature_dict = extractor.process_library(library)
        with open(os.path.join(args.directory, args.feature), "wb") as f:
            pickle.dump(feature_dict, f)

    sample = sample_from_unit_cube(feature_dict, 1000)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for feature in sample:
        ax.scatter(feature[0], feature[1], feature[2], c=feature, s=2)

    plt.show()

    for j in range(20):
        for i, feature in enumerate(sample):
            som.update(feature.cuda())
            plt.imsave('som_%i_%i' % (j, i), som.grid)


    plt.imshow(som.grid)
    plt.show()

    i = 5
    #

    # for i in range(20):
    #     for feature in features_tensor[sampled,:]:
    #         som.update(feature.cuda())

    # for i in range(10):
    #     for feature in tqdm(sorted(list(feature_dict.values()), key=lambda x: torch.norm(x-x.mean()), reverse=True), unit='image'):
    #         som.update(feature.cuda())
    # plt.imsave('som_%i' % i, som.grid/ som.grid.max())

    # plt.imshow(som.grid)
    # plt.show()









    # random.shuffle(photos)


    # plt.imshow(som.grid)
    # plt.show()

    # # returns the row / column indicies of the best matching unit for a given datapoint
    # def getPhotoCoord(val, grid, occupancy):
    #     idx = np.linalg.norm((grid + occupancy[:,:,np.newaxis] * 10000) - val, axis = 2).argmin()
    #     return idx / grid.shape[1], idx % grid.shape[1]

    # def getPhotoMatch(val, grid, occupancy):
    #     return np.linalg.norm(val - grid[getPhotoCoord(val,grid,occupancy)])
    # patch = Patchworker(som.grid.shape[:2])

    # for photo in photos:
    #     photo.coords = []
    #     photo.shape = []
    #     photo.placed = False

    # while not patch.full():

    #     # find the photo which best matches it's BMU
    #     unusedPhotos = [photo for photo in photos if photo.placed == False]

    #     if not len(unusedPhotos):
    #         for photo in photos:
    #             photo.placed = False
    #         continue

    #     idMin = np.array([getPhotoMatch(photo.val, som.grid, patch.occupied) for photo in unusedPhotos]).argmin()
    #     BMP = unusedPhotos[idMin]

    #     targetCoords = getPhotoCoord(BMP.val, som.grid, patch.occupied)
    #     coords, shape = patch.addPatch(targetCoords, BMP.footprints)
    #     BMP.coords.append(coords)
    #     BMP.shape.append(shape)
    #     BMP.placed = True

    # colors = []; coords = []

    # for photo in photos:
    #     for coord in photo.coords:
    #         coords.append(coord)

    #     for coord in photo.coords:
    #         colors.append(photo.val)

    # coordArr = np.array(coords).astype(float)

    # plt.xlim(-1, som.grid.shape[1])
    # plt.ylim(-1, som.grid.shape[0])
    # plt.scatter(coordArr[:,1], coordArr[:,0],color=colors,s=100)
    # plt.show()

    # plt.imshow(patch.occupied,interpolation='None')
    # plt.show()




    # border = 20
    # magnify = 200
    # canvas = PIL.Image.new("RGB", np.multiply(som.grid.shape[:2], magnify) + [border,border], "white")

    # for photo in photos:
    #     img = PIL.Image.open(photo.filepath)

    #     for i in range(len(photo.coords)):
    #         sizeTup = tuple(np.multiply(photo.shape[i], magnify) - border)
    #         coordTup = tuple(photo.coords[i] * magnify + border)

    #         # crop the image so that it matches its shape
    #         centroid = np.divide(img.size, 2)
    #         scale = min(img.size[0] / photo.shape[i][0], img.size[1] / photo.shape[i][1])
    #         dims = np.multiply(photo.shape[i], scale)
    #         box = (centroid[0] - dims[0] / 2, centroid[1] - dims[1] / 2, centroid[0] + dims[0] / 2, centroid[1] + dims[1] / 2)

    #         canvas.paste(img.crop(box).resize(sizeTup,PIL.Image.NEAREST), coordTup)

    # plt.imshow(canvas)
    # plt.show()



    # canvas.save('invite.png')
