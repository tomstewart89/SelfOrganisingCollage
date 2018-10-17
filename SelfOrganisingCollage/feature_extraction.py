import json
import torch
from torchvision import models, transforms
from tqdm import tqdm
from photo_library import PhotoLibrary
import matplotlib.pyplot as plt
from utils import get_imagenet_class_labels
import numpy as np
from itertools import compress

class FeatureExtractor:
    def process_library(self, library):
        feature_dict = {}
        for file_name, img in tqdm(library, unit='image'):
            feature_dict[file_name] = self.process_img(img)

        return feature_dict

    def process_img(self, img):
        raise NotImplementedError

    def draw_interesting_samples(self, feature_dict, N):
        '''Should take a dictionary containing features and return a tensor of concatenated features along with a list of file names corresponding to each feature'''
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
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])

    def process_img(self, img):
        preprocessed_img = self.preprocess(img)
        return preprocessed_img.reshape(3,-1).mean(dim=1)

    def draw_interesting_samples(self, feature_dict, N):
        '''Choose without replacement N samples randomly from the dataset that are closest to a randomly generated point on the unit cube'''

        filenames = list(feature_dict.keys())
        features = torch.stack(list(feature_dict.values()))
        sampled = torch.zeros(len(features), dtype=torch.uint8)

        for _ in range(N):
            # generate a point on the unit cube
            point = torch.rand(features.shape[1])
            point[0] = point[0].round()
            point = point[torch.randperm(features.shape[1])]

            dist_from_point = (features - point).norm(dim=1)
            dist_from_point[sampled] = dist_from_point.max()

            sampled[dist_from_point.argmin()] = 1

        return features[sampled], list(compress(filenames, sampled))

    @property
    def feature_dim(self):
        return 3


class ResNet(FeatureExtractor):
    def __init__(self, n_classes=40):
        super(ResNet, self).__init__()
    
        with open('imagenet_class_index.json') as f:
            labels_dict = json.load(f)

        self.class_labels = [labels_dict[str(i)][1] for i in range(len(labels_dict)) ]
        self.n_classes = n_classes

        self.resnet = models.resnet101(pretrained=True)
        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.activation = torch.nn.Softmax()

    def process_img(self, img):
        preprocessed_img = self.preprocess(img)
        scores = self.activation(self.resnet(preprocessed_img.unsqueeze(0)))
        sorted_scores, idx = scores.sort(descending=True)
        return sorted_scores[0,:self.n_classes], idx[0,:self.n_classes]

    def draw_interesting_samples(self, feature_dict, N):
        '''Choose samples with the highest score on any class'''

        # gather up some stuff into tensors for convenience later
        filenames = list(feature_dict.keys())
        scores = torch.stack([val[0] for val in feature_dict.values()])
        categories = torch.stack([val[1] for val in feature_dict.values()])

        # find the samples with the highest scores for their highest scoring categories
        top_scoring_sample_idx = scores.max(dim=1)[0].sort(descending=True)[1][:N]

        # isolate just the top scoring N samples
        scores_sample = scores[top_scoring_sample_idx]
        categories_sample = categories[top_scoring_sample_idx]
        
        # build a sparse float tensor to represent the huge Nx1000 feature vector
        row_idx = torch.range(0,N-1).unsqueeze(1).expand(N, categories_sample.shape[1]).long()
        features = torch.sparse.FloatTensor(torch.stack([row_idx.reshape(-1), categories_sample.reshape(-1)],dim=0), scores_sample.reshape(-1), torch.Size([N, self.feature_dim]))

        return features, [filenames[idx] for idx in top_scoring_sample_idx]

    @property
    def feature_dim(self):
        return len(self.class_labels)


if __name__ == '__main__':
    dset = PhotoLibrary('/home/tom/Pictures')
    extractor = ResNet()

    feature_dict = {}
    for i, (filename, img) in enumerate(dset):
        feature_dict[filename] = extractor.process_img(img)

        if i == 50:
            break

    extractor.draw_interesting_samples(feature_dict, 25)
