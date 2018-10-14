import random
import matplotlib.pyplot as plt
import itertools
import torch


class Patchworker:
    def __init__(self, shape):
        self.occupied = torch.zeros(shape, dtype=torch.uint8)
        self.shape = torch.Tensor(shape).int()

    def full(self):
        return self.occupied.all()

    def gen_offsets(self, shape):
        offsets = list(itertools.product(range(-shape[0]/2, shape[0]/2+1), range(-shape[1]/2, shape[1]/2+1)))
        return sorted(offsets, key=lambda x: abs(x[0]) + abs(x[1]))

    def gen_shapes(self, shape, step=1):
        aspect = float(shape[0]) / float(shape[1])

        for s in reversed(range(1, shape.min()+1)):
            yield (s, round(s / aspect)) if aspect < 1. else (round(s * aspect), s)

        yield (1,1) # this will be yielded twice in some circumstances, but it really doesn't matter

    def add_patch(self, target_coord, target_shape):
        for shape in self.gen_shapes(target_shape):
            shape = torch.Tensor(shape).int()

            for offset in self.gen_offsets(shape):
                coord = target_coord.int() + torch.Tensor(offset).int()
                if self.check_fit(coord, shape):
                    self.occupy(coord, shape)
                    return coord, shape

        # add() expects coord parameter to be unoccupied and gen_shapes will always return 1,1
        # therefore it should never be possible that the for loop above executes completely
        raise ValueError

    def check_fit(self, coord, shape):
        tl = coord - shape / 2
        br = tl + shape
        return ~self.occupied[tl[0]:br[0],tl[1]:br[1]].any() and ~(tl < torch.zeros(2).int()).any() and ~(br >= self.shape+1).any()

    def occupy(self, coord, shape):
        tl = coord - shape / 2
        br = tl + shape
        self.occupied[tl[0]:br[0],tl[1]:br[1]] = 1


if __name__ == '__main__':
    patch = Patchworker((100,200))
    canvas = torch.zeros((100,200))
    i = 0

    while not patch.full():
        idx = patch.occupied.argmin()
        target_coord = torch.Tensor([idx / patch.occupied.shape[1], idx % patch.occupied.shape[1]]).int()
        target_shape = torch.randint(5,16, (1,2), dtype=torch.int).squeeze()

        coord, shape = patch.add_patch(target_coord=target_coord, target_shape=target_shape)

        tl = coord - shape / 2
        br = tl + shape

        canvas[tl[0]:br[0],tl[1]:br[1]] = random.randint(3,10)
        canvas[coord[0],coord[1]] = 20

        i += 1
        plt.imsave('canvas_fitting/canvas_%i' % i, canvas)
