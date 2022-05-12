import numpy as np

class FeatureLoader():
    def __init__(self, path, n_features, n_arms):
        self.path = path
        self.n_features = n_features
        self.n_arms = n_arms
        
        self.load_features()
        
    # def features(self, day):
    #     features = np.load(f"{self.path}_{day}")
    #     arms = features.shape[0]
    #     if arms < self.n_arms:
    #         idx = np.random.randint(arms, size=self.n_arms-arms)
    #         features = np.concatenate((features, features[idx, :]), axis=0)
    #     elif arms > self.n_arms:
