from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis.market_analyzer import StylizedFacts, MarketVisualizer
from scipy.stats import wasserstein_distance


def aggregate_losses(loss_list):
    aggregate_loss = {'auto_correlation_loss': 0,
                      'volatility_clustering_loss': 0,
                      'leverage_effect_loss': 0,
                      'distribution_loss': 0,
                      'total_loss': 0}

    n = len(loss_list)
    for loss in loss_list:
        aggregate_loss['auto_correlation_loss'] += loss.auto_correlation_loss / n
        aggregate_loss['volatility_clustering_loss'] += loss.volatility_clustering_loss / n
        aggregate_loss['leverage_effect_loss'] += loss.leverage_effect_loss / n
        aggregate_loss['distribution_loss'] += loss.distribution_loss / n
        aggregate_loss['total_loss'] += loss.total_loss / n

    return aggregate_loss


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
        total_loss += self.distribution_loss
        total_loss /= 4
        self.total_loss = total_loss
        return total_loss

    def compute_auto_correlation_loss(self):
        target = self.target_facts.auto_correlation
        simulation = self.simulated_facts.auto_correlation
        loss = np.abs(target.values - simulation.values).mean()
        self.auto_correlation_loss = loss

    def compute_volatility_clustering_loss(self):
        target = self.target_facts.volatility_clustering
        simulation = self.simulated_facts.volatility_clustering
        loss = np.abs(target.values - simulation.values).mean()
        self.volatility_clustering_loss = loss

    def compute_leverage_effect_loss(self):
        target = self.target_facts.leverage_effect
        simulation = self.simulated_facts.leverage_effect
        loss = np.abs(target.values - simulation.values).mean()
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

    def to_df(self):
        columns = ["auto_correlation_loss", "volatility_clustering_loss",
                   "leverage_effect_loss", "distribution_loss", "total_loss"]
        values = [[self.auto_correlation_loss, self.volatility_clustering_loss,
                   self.leverage_effect_loss, self.distribution_loss, self.total_loss]]

        return pd.DataFrame(values, columns=columns)


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


def compute_total_loss(price_series, name_simulator: str):
    simulated_market_visualizer = MarketVisualizer(price_series, is_simulated=True)

    headers = ['Open', 'High', 'Low', 'Close']
    data = pd.read_csv('../data/spx/SPX_1min.txt', header=None, index_col=0, parse_dates=[0])
    data = data.drop(columns=[5])
    data.columns = headers
    real_market_visualizer = MarketVisualizer(data)

    rets_int = [1, 5, 15, 30, "1d"]
    losses = []
    t_losses = []
    for ret in rets_int:

        if ret == '1d':
            loss = LossFunction(real_market_visualizer.market_analyzer.get_daily_market_metrics(),
                                simulated_market_visualizer.market_analyzer.get_daily_market_metrics())
        else:
            loss = LossFunction(real_market_visualizer.market_analyzer.get_market_metrics(ret),
                                simulated_market_visualizer.market_analyzer.get_market_metrics(ret))
        loss.compute_loss()
        losses += [loss]
        t_losses += [loss.total_loss]

    correlation_loss = mean_absolute_error(real_market_visualizer.market_analyzer.get_close_auto_correlation(),
                                           simulated_market_visualizer.market_analyzer.get_close_auto_correlation())

    rets_int += ['close']
    t_losses += [correlation_loss]

    total_loss = aggregate_losses(losses)

    final_loss = correlation_loss + total_loss["total_loss"] * 5
    labels = ["$l_{" + str(x) + "}$" for x in rets_int]
    labels += ["$\\frac{L}{6}$"]
    t_losses += [final_loss / 6]
    colors = ['blue', 'green', 'orange', 'purple', 'red', 'brown', 'yellow']

    plt.barh(list(range(len(t_losses))), t_losses, tick_label=labels, color=colors)
    plt.title(f'Losses per Time Horizons for {name_simulator}')
    plt.xlabel('Loss')
    plt.grid(True)
    plt.show()

    print(name_simulator)
    print(labels)
    print(t_losses)

    return final_loss


def mean_absolute_error(target, simulated):
    return (np.abs(target - simulated)).mean()
