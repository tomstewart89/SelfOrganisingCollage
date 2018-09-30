photos = []

directory = os.path.join(os.getcwd(),'preprocessed')

for f in os.listdir(directory):
    filePath = os.path.join(directory, f)
    photos.append(Photo(filePath))











    som = SelfOrganisingMap(50,35,3,50,10) #(50,35,3,50,20) # (9,12,3,3,5)

random.shuffle(photos)

for photo in photos:
    som.update(photo.val)

plt.imshow(som.grid)
plt.show()

# returns the row / column indicies of the best matching unit for a given datapoint
def getPhotoCoord(val, grid, occupancy):
    idx = np.linalg.norm((grid + occupancy[:,:,np.newaxis] * 10000) - val, axis = 2).argmin()
    return idx / grid.shape[1], idx % grid.shape[1]

def getPhotoMatch(val, grid, occupancy):
    return np.linalg.norm(val - grid[getPhotoCoord(val,grid,occupancy)])
patch = Patchworker(som.grid.shape[:2])

for photo in photos:
    photo.coords = []
    photo.shape = []
    photo.placed = False

while not patch.full():

    # find the photo which best matches it's BMU
    unusedPhotos = [photo for photo in photos if photo.placed == False]

    if not len(unusedPhotos):
        for photo in photos:
            photo.placed = False
        continue

    idMin = np.array([getPhotoMatch(photo.val, som.grid, patch.occupied) for photo in unusedPhotos]).argmin()
    BMP = unusedPhotos[idMin]

    targetCoords = getPhotoCoord(BMP.val, som.grid, patch.occupied)
    coords, shape = patch.addPatch(targetCoords, BMP.footprints)
    BMP.coords.append(coords)
    BMP.shape.append(shape)
    BMP.placed = True

colors = []; coords = []

for photo in photos:
    for coord in photo.coords:
        coords.append(coord)

    for coord in photo.coords:
        colors.append(photo.val)

coordArr = np.array(coords).astype(float)

plt.xlim(-1, som.grid.shape[1])
plt.ylim(-1, som.grid.shape[0])
plt.scatter(coordArr[:,1], coordArr[:,0],color=colors,s=100)
plt.show()

plt.imshow(patch.occupied,interpolation='None')
plt.show()




border = 20
magnify = 200
canvas = PIL.Image.new("RGB", np.multiply(som.grid.shape[:2], magnify) + [border,border], "white")

for photo in photos:
    img = PIL.Image.open(photo.filepath)

    for i in range(len(photo.coords)):
        sizeTup = tuple(np.multiply(photo.shape[i], magnify) - border)
        coordTup = tuple(photo.coords[i] * magnify + border)

        # crop the image so that it matches its shape
        centroid = np.divide(img.size, 2)
        scale = min(img.size[0] / photo.shape[i][0], img.size[1] / photo.shape[i][1])
        dims = np.multiply(photo.shape[i], scale)
        box = (centroid[0] - dims[0] / 2, centroid[1] - dims[1] / 2, centroid[0] + dims[0] / 2, centroid[1] + dims[1] / 2)

        canvas.paste(img.crop(box).resize(sizeTup,PIL.Image.NEAREST), coordTup)

plt.imshow(canvas)
plt.show()



canvas.save('invite.png')
