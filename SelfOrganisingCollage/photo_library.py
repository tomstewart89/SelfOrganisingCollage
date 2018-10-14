import os
import glob
from PIL import Image, ImageEnhance
from utils import flatten
import matplotlib.pyplot as plt


IMG_EXTENSIONS = [f(x) for x in ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif'] for f in (str.upper, str.lower)]


class PhotoLibrary:
    def __init__(self, directory, enhance=3.0):
        self.img_files = flatten([glob.glob(os.path.join(directory, '**/*'+ext), recursive=True) for ext in IMG_EXTENSIONS])
        self.enhance = enhance

    def __len__(self):
        return len(self.img_files)

    def __iter__(self):
        for img_file in self.img_files:
            img = Image.open(img_file).convert(mode='RGB')
            converter = ImageEnhance.Color(img)
            yield img_file, converter.enhance(self.enhance)


if __name__ == '__main__':
    dset = PhotoLibrary('/home/tom/Pictures')

    for file_name, img in dset:
        plt.imshow(img)
        plt.show()




# class Photo:
#     def __init__(self, filepath):

#         img = misc.imread(filepath)

#         self.filepath = filepath

#         self.coords = []
#         self.shape = []
#         self.placed = False

#         self.originalSize = img.shape
#         self.val = self.extractMean(img)
#         self.mean = self.extractMean(img)
#         self.fp = [[1,1],[1,2],[2,2],[2,3],[3,3],[3,4]]



#     @property
#     def footprints(self):

#         fp = np.sort(self.fp)
#         ids = np.argsort(np.prod(fp,axis=1))
#         fp = fp[ids][::-1]

#         if self.originalSize[1] > self.originalSize[0]:
#             fp = np.fliplr(fp)

#         return fp