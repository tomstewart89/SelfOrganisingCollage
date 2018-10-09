from torchvision import models, transforms
from tqdm import tqdm
from photo_library import PhotoLibrary



class FeatureExtractor:
    def process_library(self, library):
        feature_dict = {}
        for file_name, img in tqdm(library, unit='image'):
            feature_dict[file_name] = self.process_img(img)
        
        return feature_dict

    def process_img(self, img):
        raise NotImplementedError

    @property
    def feature_dim(self):
        raise NotImplementedError

    @staticmethod
    def factory(type):
        if type == "mean_color": 
            return MeanColor()
        elif type == "resnet": 
            return ResNet()
        else:
            raise KeyError

class MeanColor(FeatureExtractor):
    def __init__(self):
        super(MeanColor, self).__init__()
        self.preprocess = transforms.ToTensor()

    def process_img(self, img):
        preprocessed_img = self.preprocess(img)
        return preprocessed_img.reshape(3,-1).mean(dim=1)

    @property
    def feature_dim(self):
        return 3


class ResNet(FeatureExtractor):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = models.resnet101(pretrained=True)
        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def process_img(self, img):
        preprocessed_img = self.preprocess(img)
        return self.resnet(preprocessed_img.unsqueeze(0))

    @property
    def feature_dim(self):
        return 1000


if __name__ == '__main__':
    dset = PhotoLibrary('/home/tom/Pictures')
    it = iter(dset)

    extrator = ResNet()

    print(extrator.process_img(next(it)[1]))
