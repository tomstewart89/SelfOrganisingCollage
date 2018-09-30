class Photo:
    def __init__(self, filepath):

        img = misc.imread(filepath)

        self.filepath = filepath

        self.coords = []
        self.shape = []
        self.placed = False

        self.originalSize = img.shape
        self.val = self.extractMean(img)
        self.mean = self.extractMean(img)
        self.fp = [[1,1],[1,2],[2,2],[2,3],[3,3],[3,4]]

    def extractMean(self, img):
        val = img.mean(axis=(0,1))[:3]
        val /= 255
        return val

    def extractLargestCluster(self, img):
        img = img[:3]
        # reshape the array and take a random sample from it
        image_array = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
        image_array_sample = shuffle(image_array, random_state=0)[:1000] / 255.

        # cluster the sampled pixels into 64 groups
        kmeans = KMeans(n_clusters = 64, random_state = 0).fit(image_array_sample)

        # find the largest group and return it
        _ , counts = np.unique(kmeans.labels_, return_counts=True)
        idx = np.argsort(counts)[::-1]
        return kmeans.cluster_centers_[idx[0]]

    @property
    def footprints(self):

        fp = np.sort(self.fp)
        ids = np.argsort(np.prod(fp,axis=1))
        fp = fp[ids][::-1]

        if self.originalSize[1] > self.originalSize[0]:
            fp = np.fliplr(fp)

        return fp
