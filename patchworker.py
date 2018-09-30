class Patchworker:
    def __init__(self, size):
        self.occupied = np.zeros(size)

    def reset(self):
        self.occupied.fill(0)

    def full(self):
        return np.prod(patch.occupied.shape) - np.count_nonzero(patch.occupied) == 0

    def addPatch(self, coords, shapes):

        for shape in shapes:
            for rowOffset in range(1 - shape[0], 1):
                for colOffset in range(1 - shape[1], 1):

                    shiftedCoords = np.add(coords, (rowOffset, colOffset))

                    if self.checkFit(shiftedCoords, shape):
                        self.occupied[shiftedCoords[0] : shiftedCoords[0] + shape[0], shiftedCoords[1] : shiftedCoords[1] + shape[1]] = random.randint(0,255)
                        return shiftedCoords, shape


    # returns true if the patch fits in the specified spot in the patchwork
    def checkFit(self, coord, shape):

        # return false if the coord is out of bounds
        if coord[0] + shape[0] > self.occupied.shape[0] or coord[1] + shape[1] > self.occupied.shape[1] or coord[0] < 0 or coord[1] < 0:
            return False

        place = np.zeros(self.occupied.shape)
        place[coord[0] : coord[0] + shape[0], coord[1] : coord[1] + shape[1]] = 1

        return ~np.any(self.occupied * place)

if __name__ == '__main__':
    patch = Patchworker((20,20))

    while not patch.full():

        empties = np.where(patch.occupied==0)
        choice = random.randint(0,len(empties[0])-1)
        targetCoord = (empties[0][choice], empties[1][choice])

        coord, shape = patch.addPatch(targetCoord, [[3,4] if random.randint(0,1) else [4,3], [3,3], [2,2], [1,1]])

    plt.imshow(patch.occupied,interpolation='none')
    plt.show()
