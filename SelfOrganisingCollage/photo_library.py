import os
import glob
from PIL import Image, ImageEnhance
from utils import flatten, autorotate
import matplotlib.pyplot as plt


IMG_EXTENSIONS = [f(x) for x in ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif'] for f in (str.upper, str.lower)]

class PhotoLibrary:
    def __init__(self, directory, enhance=2.0):
        self.img_files = flatten([glob.glob(os.path.join(directory, '**/*'+ext), recursive=True) for ext in IMG_EXTENSIONS])
        self.enhance = enhance

    def __len__(self):
        return len(self.img_files)

    def __iter__(self):
        for img_file in self.img_files:
            yield img_file, self[img_file]

    def __getitem__(self, img_file):
        img = Image.open(img_file)
        img = autorotate(img)
        img = img.convert(mode='RGB')
        converter = ImageEnhance.Color(img)
        return converter.enhance(self.enhance)


if __name__ == '__main__':
    dset = PhotoLibrary('/home/tom/Pictures')

    for filename, img in dset:
        print(filename)
        # plt.imshow(img)
        # plt.show()
