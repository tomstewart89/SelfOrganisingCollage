import random
import matplotlib.pyplot as plt
import itertools
import torch

# ok so I want to place a square of a given shape at a given position in the grid. I can assume that the supplied coordinate is unoccupied. 
# So I want to start with the largest photo size check if its a fit and if not, reduce by one. 
# I can slide the reduced photo around as many times as I have reduced it so


class Patchworker:
    def __init__(self, shape):
        self.occupied = torch.zeros(shape, dtype=torch.uint8)
        self.shape = torch.Tensor(shape).int()

    def full(self):
        return self.occupied.all()

    def gen_shapes(self, shape, step=1):
        aspect = float(shape[0]) / float(shape[1])

        for s in reversed(range(1, shape.min()+1)):
            yield (s, round(s / aspect)) if aspect < 1. else (round(s * aspect), s)

    def add(self, coord, shape):
        for s in self.gen_shapes(shape):
            for offset in itertools.product(range(-d[0], d[0]+1), range(-d[1], d[1]+1)):
                shifted_coord = coord.int() + torch.Tensor(offset).int()

                if self.check_fit(shifted_coord, shape):
                    self.occupy(shifted_coord, shape)
                    return shifted_coord, shape

        return None, None

    def check_fit(self, coord, shape):
        tr = (coord + shape / 2)
        bl = (coord - shape / 2)
        return not(self.occupied[bl[0]:tr[0],bl[1]:tr[1]].any() or (bl < torch.zeros(2).int()).any() or (tr >= self.shape).any())

    def occupy(self, coord, shape):
        tr = (coord + shape / 2)
        bl = (coord - shape / 2)
        self.occupied[bl[0]:tr[0],bl[1]:tr[1]] = 1

    def get_unoccupied(self):
        idx = self.occupied.argmin()
        return torch.Tensor([idx / self.occupied.shape[1], idx % self.occupied.shape[1]])


def get_patch(arr, coord, shape):
    tr = (coord + shape / 2)
    bl = (coord - shape / 2)
    return arr[bl[0]:tr[0],bl[1]:tr[1]]


if __name__ == '__main__':
    patch = Patchworker((200,200))
    canvas = torch.zeros((200,200))

    for s in patch.gen_shapes(torch.Tensor([16,9]).int(),step=1):
        print(s)

    while not patch.full():
        coord = patch.get_unoccupied()
        coord, shape = patch.add(coord=coord,shape=torch.randint(32,50, (1,2), dtype=torch.int).squeeze())
        
        tr = (coord + shape / 2)
        bl = (coord - shape / 2)
        canvas[bl[0]:tr[0],bl[1]:tr[1]] = random.randint(0,10)
        

    plt.imshow(canvas,interpolation='none')
    plt.show()

    i = 6