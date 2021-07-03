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
