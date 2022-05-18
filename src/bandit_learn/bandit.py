import numpy as np
import itertools
import random
import torch
from .feature_loader import FeatureLoader
from stock.dataLoader import StockLoader
from datetime import datetime, timedelta
import dask.array as da


class ContextualBandit():
    def __init__(self,
                 T,
                 n_arms,
                 n_features,
                 h=None,
                 noise_std=1.0,
                 seed=None,
                 feature_loader: FeatureLoader=None,
                 stock_loader: StockLoader=None,
                 ):
        # if not None, freeze seed for reproducibility
        self._seed(seed)

        # number of rounds
        self.T = T
        # number of arms
        self.n_arms = n_arms
        # number of features for each arm
        self.n_features = n_features
        # average reward function
        # h : R^d -> R
        self.h = h

        # standard deviation of Gaussian reward noise
        self.noise_std = noise_std

        self.feature_loader = feature_loader
        self.stock_loader = stock_loader

        # generate random features
        self.reset()

    @property
    def arms(self):
        """Return [0, ...,n_arms-1]
        """
        return range(self.n_arms)

    def reset(self):
        """Generate new features and new rewards.
        """
        if self.feature_loader:
            self.feature_loader.loadFeatures()
            self.features = self.feature_loader.features_df["features"].values.compute_chunk_sizes()
            self.inference = self.feature_loader.features_df["inference"].values.compute_chunk_sizes()
            self.confidence = self.feature_loader.features_df["confidence"].values.compute_chunk_sizes()
            self.dates = self.feature_loader.features_df["date"].values.compute_chunk_sizes()
            self.rate = dict()
            self.rewards = da.concatenate(
                [
                    self.get_rewards(i) for i in range(self.T)
                ],
                axis=0,
            )
            self.best_rewards_oracle = da.max(self.rewards, axis=1)
            self.best_actions_oracle = da.argmax(self.rewards, axis=1)
        else:
            self.reset_features()
            self.reset_rewards()

    def get_rewards(self, index):
        date_str = self.dates[index].compute()
        date = datetime.strptime(date_str, "%Y-%m-%d")
        if date_str not in self.rate:
            previous_date = date - timedelta(1)
            try:
                self.rate[date_str] = self.stock_loader.GetRateFromPeriod(start=previous_date.strftime("%Y-%m-%d"), end=date_str)
            except KeyError:
                self.rate[date_str] = 0
            while self.rate[date_str] == 0:
                previous_date -= timedelta(1)
                try:
                    self.rate[date_str] = self.stock_loader.GetRateFromPeriod(start=previous_date.strftime("%Y-%m-%d"), end=date_str)
                except KeyError:
                    self.rate[date_str] = 0
        confidence = np.array(self.confidence[index].compute())
        inference = np.array(self.inference[index].compute())
        inference = np.array([1 if inf == "LABEL_1" else -1 for inf in inference])
        return da.from_array(inference * confidence * self.rate[date_str])

    def reset_features(self):
        """Generate normalized random N(0,1) features.
        """
        x = np.random.randn(self.T, self.n_arms, self.n_features)
        x /= np.repeat(np.linalg.norm(x, axis=-1, ord=2), self.n_features).reshape(self.T, self.n_arms, self.n_features)
        self.features = x

    def reset_rewards(self):
        """Generate rewards for each arm and each round,
        following the reward function h + Gaussian noise.
        """
        self.rewards = np.array(
            [
                self.h(self.features[t, k]) + self.noise_std*np.random.randn()
                for t, k in itertools.product(range(self.T), range(self.n_arms))
            ]
        ).reshape(self.T, self.n_arms)

        # to be used only to compute regret, NOT by the algorithm itself
        self.best_rewards_oracle = np.max(self.rewards, axis=1)
        self.best_actions_oracle = np.argmax(self.rewards, axis=1)

    def _seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
