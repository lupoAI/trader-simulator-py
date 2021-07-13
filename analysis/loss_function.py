from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis.market_analyzer import StylizedFacts
from utilities.scipy_utils import wasserstein_distance


class LossFunction:

    def __init__(self, target_facts: StylizedFacts, simulated_facts: StylizedFacts):
        self.target_facts = target_facts
        self.simulated_facts = simulated_facts
        self.auto_correlation_loss = None
        self.volatility_clustering_loss = None
        self.leverage_effect_loss = None
        self.distribution_loss = None
        self.total_loss = None

    def compute_loss(self):
        if self.auto_correlation_loss is None:
            self.compute_auto_correlation_loss()
        if self.volatility_clustering_loss is None:
            self.compute_volatility_clustering_loss()
        if self.leverage_effect_loss is None:
            self.compute_leverage_effect_loss()
        if self.distribution_loss is None:
            self.compute_distribution_loss()
        total_loss = 0
        total_loss += self.auto_correlation_loss
        total_loss += self.volatility_clustering_loss
        total_loss += self.leverage_effect_loss
        # Scale the distribution function
        total_loss += self.distribution_loss / 100
        total_loss /= 4
        self.total_loss = total_loss
        return total_loss

    def compute_auto_correlation_loss(self):
        target = self.target_facts.auto_correlation
        simulation = self.simulated_facts.auto_correlation
        loss = ((target.values - simulation.values) ** 2).mean()
        self.auto_correlation_loss = loss

    def compute_volatility_clustering_loss(self):
        target = self.target_facts.volatility_clustering
        simulation = self.simulated_facts.volatility_clustering
        loss = ((target.values - simulation.values) ** 2).mean()
        self.volatility_clustering_loss = loss

    def compute_leverage_effect_loss(self):
        target = self.target_facts.leverage_effect
        simulation = self.simulated_facts.leverage_effect
        loss = ((target.values - simulation.values) ** 2).mean()
        self.leverage_effect_loss = loss

    def compute_distribution_loss(self):
        # target = self.target_facts.density
        # simulation = self.simulated_facts.density
        # loss = wasserstein_distance(target.index.values, simulation.index.values,
        #                             target.values, simulation.values)
        target = self.target_facts.rets
        simulation = self.simulated_facts.rets
        loss = wasserstein_distance(target, simulation)
        self.distribution_loss = loss


class LossAnalyzer:

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.results_df = pd.read_csv(file_path, index_col=None)

    def visualize_relationship(self, independent_variable: str, dependent_variable: str, save_name=None):
        ind_var = self.results_df[independent_variable]
        dep_var = self.results_df[dependent_variable]

        lr = LinearRegression()
        lr.fit(ind_var.values.reshape(-1, 1), dep_var.values)
        min_ind = ind_var.min()
        max_ind = ind_var.max()
        range_pred = np.arange(min_ind, max_ind, (max_ind - min_ind) / 100)
        y_pred = lr.predict(range_pred.reshape(-1, 1))
        plt.scatter(ind_var, dep_var, label='scatter')
        plt.plot(range_pred, y_pred, label='linear fit', color='red')
        plt.xlabel(independent_variable)
        plt.ylabel(dependent_variable)
        plt.legend()
        if save_name is not None:
            plt.savefig(save_name)
        plt.show()
