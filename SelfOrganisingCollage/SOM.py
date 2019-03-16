import math as m
import matplotlib.pyplot as plt
import torch
import cupy as cp
from utils import TimeIt
from tqdm import tqdm


class SelfOrganisingMap:
    def __init__(self, shape, sigma=5., eta=5.):
        self.grid = cp.random.rand(*shape)
        self.sigma = sigma
        self.eta = eta

        xx, yy = cp.meshgrid(cp.arange(shape[1]), cp.arange(shape[0]))
        self.elem_idx = cp.stack([yy,xx], axis=2)

    # returns the row / column indicies of the best matching unit for a given datapoint
    def get_BMU(self, x):
        idx = cp.argmin(cp.linalg.norm(self.grid - cp.array(x), axis=2))
        return cp.unravel_index(idx, self.grid.shape[:2])

    def get_BMU_dist(self, x):
        return cp.linalg.norm(x-self.grid[self.get_BMU(x)])

    # returns an array representing the distance of each cell from the given row/col
    def dist(self, row, col):
        idx = cp.stack([row, col])
        return cp.linalg.norm(cp.abs(self.elem_idx - idx), axis=2)

    def gauss(self, x, mu, sigma):
        return 1. / m.sqrt(2. * sigma * m.pi) * cp.exp(-cp.power((x-mu),2) / (2.*sigma))

    def update(self, x):
        # get the best matching unit
        r,c = self.get_BMU(x)

        # now see how much we need to correct all the units around the BMU
        alpha = self.gauss(self.dist(r,c), 0, self.sigma)

        # now calculate how much to add to each of the neighbouring units
        delta = cp.expand_dims(alpha,2) * (x - self.grid)

        # lastly update the grid
        self.grid += (delta * self.eta)


if __name__ == '__main__':
    som = SelfOrganisingMap(shape=[35,50,3], sigma=5., eta=1.)

    x = [0.9, 0.4, 0.0]

    r,c = som.get_BMU(x)
    alpha = som.gauss(som.dist(r,c),0,som.sigma)
    delta = cp.expand_dims(alpha,2) * (cp.array(x) - som.grid)
    updated = som.grid + delta * som.eta

    # Show the update process
    for i, (title, img) in enumerate(zip(['original', 'dist', 'alpha', 'delta', 'updated'], [som.grid, som.dist(r,c), alpha, delta, updated])):
        plt.subplot(1, 5, i+1)
        plt.title(title)
        plt.imshow(cp.asnumpy(img),interpolation='none')

    plt.show()

    # Show an updated map
    for i in tqdm(range(10000), unit='feature'):
        som.update(cp.random.rand(3))
        plt.imsave('som_training_1/som_%i' % i, cp.asnumpy(som.grid))

    plt.imshow(cp.asnumpy(som.grid),interpolation='none')
    plt.show()
