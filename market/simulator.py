from math import gamma

import numpy as np
from numpy import exp, array, log, power, cumsum
from numpy.random import RandomState

from market.agent import Agent, AgentFCN, AgentGamma
from market.data_model import MarketSnapshotSeries, MarketSnapshot
from market.data_model import Side, Series
from market.exchange import Exchange

MINUTES_IN_DAY = 390


class Simulator:

    def __init__(self, exchange: Exchange, n_agents: int):
        self.exchange = exchange
        self.n_agents = n_agents
        self.market_snapshots = MarketSnapshotSeries()
        self.mid_price_series = Series()
        self.last_mid_price_series = Series()

    def __del__(self):
        del self.exchange
        del self.n_agents
        del self.market_snapshots
        del self.mid_price_series
        del self.last_mid_price_series
        del self

    def save_market_snapshot(self, time_step: int):
        market_snapshot = self.exchange.return_market_snapshot()
        self.market_snapshots.add(MarketSnapshot(time_step=time_step, **market_snapshot))

    def clear_cache(self):
        self.market_snapshots = MarketSnapshotSeries()
        self.mid_price_series = Series()


class RandomSimulator(Simulator):

    def __init__(self, exchange: Exchange, n_agents: int):
        super().__init__(exchange, n_agents)
        self.agents = [Agent(self.exchange) for _ in range(self.n_agents)]

    def run(self, n_steps: int, trades_per_step: int, starting_price: int, snapshot_interval: int,
            cancel_order_interval: int, random_seed: int = 42):
        self.clear_cache()
        rand_state = RandomState(random_seed)
        size_matrix = (n_steps, trades_per_step)
        agents = rand_state.randint(0, self.n_agents, size=size_matrix)
        sides = rand_state.randint(1, 3, size=size_matrix)
        volume = 1
        vol = 0.02
        mu = 0
        vol_adjustment = -vol ** 2 / 2
        offset = rand_state.normal(0, vol, size=size_matrix)
        multiplier = np.exp(offset + vol_adjustment + mu)
        orders_to_cancel = {}
        for i in range(n_steps):
            if i % snapshot_interval == 0 and i > 0:
                self.save_market_snapshot(i)

            orders_to_cancel[i] = []
            for j in range(trades_per_step):
                if self.exchange.last_valid_mid_price is None:
                    price = int(starting_price * multiplier[i, j] + 0.5)
                else:
                    price = int(self.exchange.last_valid_mid_price * multiplier[i, j] + 0.5)
                order_receipt = self.agents[agents[i, j]].limit_order(Side(sides[i, j]), price, volume, True)
                orders_to_cancel[i] += [(agents[i, j], order_receipt.order_id)]

            self.mid_price_series.add(i, self.exchange.mid_price)
            self.last_mid_price_series.add(i, self.exchange.last_valid_mid_price)

            if i >= cancel_order_interval:
                for agent_id, order_id in orders_to_cancel[i - cancel_order_interval]:
                    self.agents[agent_id].cancel_order(order_id)

        self.market_snapshots.format_price_to_volume()


class SimulatorFCN(Simulator):

    def __init__(self, exchange: Exchange, n_agents: int, initial_fund_price: int, fund_price_vol: float,
                 scale_fund=None, scale_chart=None, scale_noise=None, fund_price_trend=0, random_seed: int = 42):
        super().__init__(exchange, n_agents)
        self.fund_price_vol = fund_price_vol
        self.fund_price_trend = fund_price_trend
        self.initial_fund_price = initial_fund_price
        rand_state = RandomState(random_seed)
        if scale_fund is not None:
            self.scale_fund = scale_fund
        else:
            self.scale_fund = rand_state.normal(0.3, 0.03)
        if scale_chart is not None:
            self.scale_chart = scale_chart
        else:
            self.scale_chart = rand_state.normal(0.3, 0.03)
        if scale_noise is not None:
            self.scale_noise = scale_noise
        else:
            self.scale_noise = rand_state.normal(0.3, 0.03)
        self.agents_fcn = rand_state.exponential(size=(n_agents, 3))
        self.agents_fcn *= array([[self.scale_fund, self.scale_chart, self.scale_noise]])
        self.agents_time_window = rand_state.randint(500, 1001, size=(n_agents,))
        self.agents_order_margin = rand_state.uniform(0, 0.05, size=(n_agents,))
        self.agents = [AgentFCN(self.exchange,
                                *self.agents_fcn[i],
                                self.agents_time_window[i],
                                self.agents_order_margin[i]) for i in range(self.n_agents)]
        self.fund_price_series = array([])

    def create_fundamental_price_series(self, n_steps, random_seed: int):
        rand_state = RandomState(random_seed)
        evolution = rand_state.normal(size=n_steps)
        percentage_change = exp(
            evolution * self.fund_price_vol + self.fund_price_trend - 0.5 * self.fund_price_vol ** 2)
        time_series = percentage_change.cumprod() * self.initial_fund_price
        self.fund_price_series = time_series

    def clear_cache(self):
        super().clear_cache()
        self.fund_price_series = array([])

    def run(self, n_steps: int, trades_per_step: int, snapshot_interval: int,
            cancel_order_interval: int, random_seed: int = 42):
        self.clear_cache()
        rand_state = RandomState(random_seed)
        self.create_fundamental_price_series(n_steps, rand_state.randint(1000))
        size_matrix = (n_steps, trades_per_step)
        agents = rand_state.randint(0, self.n_agents, size=size_matrix)
        agents_noise = rand_state.normal(0, 0.0001, size=size_matrix)
        agents_volume = rand_state.randint(1, 5, size=size_matrix)
        agents_buy_outcome = rand_state.uniform(size=size_matrix)
        orders_to_cancel = {}
        for i in range(n_steps):
            if i % snapshot_interval == 0 and i > 0:
                self.save_market_snapshot(i)

            orders_to_cancel[i] = []
            for j in range(trades_per_step):
                time_window = self.agents[agents[i, j]].submit_time_window()
                if time_window > i - 1:
                    previous_price = None
                else:
                    previous_price = self.last_mid_price_series[-time_window]
                self.agents[agents[i, j]].get_data(self.fund_price_series[i], self.initial_fund_price, previous_price,
                                                   agents_noise[i, j])
                order_receipt = self.agents[agents[i, j]].decide_order(agents_volume[i, j], agents_buy_outcome[i, j])
                orders_to_cancel[i] += [(agents[i, j], order_receipt.order_id)]

            self.mid_price_series.add(i, self.exchange.mid_price)
            self.last_mid_price_series.add(i, self.exchange.last_valid_mid_price)

            if i >= cancel_order_interval:
                for agent_id, order_id in orders_to_cancel[i - cancel_order_interval]:
                    self.agents[agent_id].cancel_order(order_id)
        self.market_snapshots.format_price_to_volume()


class SimulatorPaper1(Simulator):

    def __init__(self, exchange: Exchange, n_agents: int, starting_price: int, starting_cash: int, starting_stock: int):
        super().__init__(exchange, n_agents)
        self.starting_price = starting_price
        self.starting_cash = starting_cash
        self.starting_stock = starting_stock
        self.agents = [Agent(self.exchange) for _ in range(self.n_agents)]
        for agent in self.agents:
            agent.endow(starting_cash, starting_stock)

    def run(self, time_simulation: int, snapshot_interval: int, mean_wait_time: int, margin_vol: float, beta: float,
            lifespan_order: int, random_seed: int = 42):
        super().clear_cache()
        rand_state = RandomState(random_seed)
        number_draws = 2 * int(time_simulation / mean_wait_time)
        rand_uniform = rand_state.uniform(size=number_draws)
        eta = mean_wait_time / gamma(1 / beta + 1)
        weibull_waiting_times = eta * power(-log(rand_uniform), 1 / beta)
        weibull_order_times = cumsum(weibull_waiting_times)
        if weibull_order_times[-1] < time_simulation:
            raise ValueError(f"Not enough time order simulation: {weibull_order_times[-1]}")
        weibull_order_times = (weibull_order_times[weibull_order_times <= time_simulation] + 0.5).astype(int)
        number_trades = len(weibull_order_times)
        chosen_agents = rand_state.randint(0, len(self.agents), size=number_trades)
        chosen_side = rand_state.randint(1, 3, size=number_trades)
        chosen_margin = rand_state.normal(1, margin_vol, size=number_trades)
        chosen_quantity = rand_state.uniform(size=number_trades)
        orders = [(x, "O", i) for i, x in enumerate(weibull_order_times)]
        cancellations = [(x + lifespan_order, "C", i) for i, x in enumerate(weibull_order_times)]
        actions = sorted(orders + cancellations)

        order_id_to_exchange_order_id = {}
        previous_time = 0
        for action in actions:

            time = action[0]
            type_action = action[1]
            idx = action[2]

            if time // snapshot_interval > previous_time // snapshot_interval:
                self.save_market_snapshot(time)
            previous_time = time

            agent_id = chosen_agents[idx]
            side = chosen_side[idx]
            margin = chosen_margin[idx]
            quantity = chosen_quantity[idx]
            if type_action == "C":
                order_id = order_id_to_exchange_order_id[idx]
                self.agents[agent_id].cancel_order(order_id)
            elif type_action == "O":

                if side == 1:
                    best_bid = self.exchange.best_bid_price
                    if self.exchange.last_valid_mid_price is not None:
                        best_bid = best_bid if best_bid is not None else int(
                            self.exchange.last_valid_mid_price * 0.995 + 0.5)
                    best_bid = best_bid if best_bid is not None else int(self.starting_price * 0.995 + 0.5)
                    order_volume = (quantity * self.agents[agent_id].cash / best_bid + 0.5)
                    order_receipt = self.agents[agent_id].limit_order(Side.BUY, int(0.9975 * best_bid * margin + 0.5),
                                                                      order_volume, True)
                else:
                    best_ask = self.exchange.best_ask_price
                    if self.exchange.last_valid_mid_price is not None:
                        best_ask = best_ask if best_ask is not None else int(
                            self.exchange.last_valid_mid_price * 1.005 + 0.5)
                    best_ask = best_ask if best_ask is not None else int(self.starting_price * 1.005 + 0.5)
                    order_volume = int(quantity * self.agents[agent_id].stock + 0.5)
                    order_receipt = self.agents[agent_id].limit_order(Side.SELL, int(1.0025 * best_ask * margin + 0.5),
                                                                      order_volume, True)

                order_id_to_exchange_order_id[idx] = order_receipt.order_id
                self.mid_price_series.add(time, self.exchange.mid_price)
            else:
                raise ValueError("Wrong action type")
        self.market_snapshots.format_price_to_volume()


class SimulatorFCNExp(Simulator):

    def __init__(self, exchange: Exchange, n_agents: int, initial_fund_price: int, fund_price_vol: float,
                 scale_fund=None, scale_chart=None, scale_noise=None, fund_price_trend=0, random_seed: int = 42):
        super().__init__(exchange, n_agents)
        self.fund_price_vol = fund_price_vol
        self.fund_price_trend = fund_price_trend
        self.initial_fund_price = initial_fund_price
        rand_state = RandomState(random_seed)
        if scale_fund is not None:
            self.scale_fund = scale_fund
        else:
            self.scale_fund = rand_state.normal(0.3, 0.03)
        if scale_chart is not None:
            self.scale_chart = scale_chart
        else:
            self.scale_chart = rand_state.normal(0.3, 0.03)
        if scale_noise is not None:
            self.scale_noise = scale_noise
        else:
            self.scale_noise = rand_state.normal(0.3, 0.03)
        self.agents_fcn = rand_state.exponential(size=(n_agents, 3))
        self.agents_fcn *= array([[self.scale_fund, self.scale_chart, self.scale_noise]])
        self.agents_time_window = rand_state.randint(15, 700, size=(n_agents,))
        self.agents_order_margin = rand_state.uniform(0, 0.01, size=(n_agents,))
        self.agents = [AgentFCN(self.exchange,
                                *self.agents_fcn[i],
                                self.agents_time_window[i],
                                self.agents_order_margin[i]) for i in range(self.n_agents)]
        self.fund_price_series = array([])

    def create_fundamental_price_series(self, n_steps, random_seed: int):
        rand_state = RandomState(random_seed)
        evolution = rand_state.normal(size=n_steps)
        percentage_change = exp(
            evolution * self.fund_price_vol + self.fund_price_trend - 0.5 * self.fund_price_vol ** 2)
        time_series = percentage_change.cumprod() * self.initial_fund_price
        self.fund_price_series = time_series

    def clear_cache(self):
        super().clear_cache()
        self.fund_price_series = array([])

    def run(self, n_steps: int, average_trades_per_step: int, snapshot_interval: int,
            cancel_order_interval: int, random_seed: int = 42):
        self.clear_cache()

        rand_state = RandomState(random_seed)

        self.create_fundamental_price_series(n_steps, rand_state.randint(1000))

        n_draws = n_steps * average_trades_per_step * 2

        waiting_times = rand_state.exponential(1 / average_trades_per_step, size=n_draws)
        time_of_trade = waiting_times.cumsum()
        minute_of_trade = time_of_trade.astype(int)
        minute_of_trade = minute_of_trade[minute_of_trade < n_steps]
        n_trades = len(minute_of_trade)

        agents = rand_state.randint(0, self.n_agents, size=n_trades)
        agents_noise = rand_state.normal(0, 0.0001, size=n_trades)
        agents_volume = rand_state.randint(1, 5, size=n_trades)
        agents_buy_outcome = rand_state.uniform(size=n_trades)
        orders_to_cancel = []
        current_minute = 0
        changed_period = True

        for i in range(n_trades):

            if current_minute != minute_of_trade[i]:
                changed_period = True
                current_minute = minute_of_trade[i]

            if current_minute % snapshot_interval == 0 and current_minute > 0 and changed_period:
                self.save_market_snapshot(current_minute)

            if changed_period:
                orders_to_cancel += [[]]

            time_window = self.agents[agents[i]].submit_time_window()
            if time_window > current_minute - 1:
                previous_price = None
            else:
                if len(self.last_mid_price_series) >= time_window:
                    previous_price = self.last_mid_price_series[-time_window]
                else:
                    previous_price = self.last_mid_price_series[0]

            self.agents[agents[i]].get_data(self.fund_price_series[current_minute], self.initial_fund_price,
                                            previous_price, agents_noise[i])
            order_receipt = self.agents[agents[i]].decide_order(agents_volume[i], agents_buy_outcome[i])
            orders_to_cancel[-1] += [(agents[i], order_receipt.order_id)]

            if changed_period:
                self.mid_price_series.add(current_minute, self.exchange.mid_price)
                self.last_mid_price_series.add(current_minute, self.exchange.last_valid_mid_price)

            if current_minute >= cancel_order_interval and changed_period:
                for agent_id, order_id in orders_to_cancel[0]:
                    self.agents[agent_id].cancel_order(order_id)
                del orders_to_cancel[0]

            changed_period = False

        self.market_snapshots.format_price_to_volume()


class SimulatorFCNGamma(Simulator):

    def __init__(self, exchange: Exchange, n_agents: int, initial_fund_price: int, fund_price_vol: float,
                 scale_fund=None, scale_chart=None, scale_noise=None, gamma_traders_percentage=0, fund_price_trend=0,
                 random_seed: int = 42):
        super().__init__(exchange, n_agents)
        self.fund_price_vol = fund_price_vol
        self.fund_price_trend = fund_price_trend
        self.initial_fund_price = initial_fund_price
        rand_state = RandomState(random_seed)
        if scale_fund is not None:
            self.scale_fund = scale_fund
        else:
            self.scale_fund = rand_state.normal(0.3, 0.03)
        if scale_chart is not None:
            self.scale_chart = scale_chart
        else:
            self.scale_chart = rand_state.normal(0.3, 0.03)
        if scale_noise is not None:
            self.scale_noise = scale_noise
        else:
            self.scale_noise = rand_state.normal(0.3, 0.03)
        self.n_gamma_agents = int(gamma_traders_percentage * n_agents)
        self.n_fcn_agents = n_agents - self.n_gamma_agents
        self.agents_fcn = rand_state.exponential(size=(self.n_fcn_agents, 3))
        self.agents_fcn *= array([[self.scale_fund, self.scale_chart, self.scale_noise]])
        self.agents_time_window = rand_state.randint(500, 1001, size=(self.n_fcn_agents,))  # Modify
        self.agents_order_margin = rand_state.uniform(0, 0.05, size=(self.n_fcn_agents,))  # Modify
        self.agents_fcn = [AgentFCN(self.exchange,
                                    *self.agents_fcn[i],
                                    self.agents_time_window[i],
                                    self.agents_order_margin[i]) for i in range(self.n_fcn_agents)]
        self.agents_gamma = [AgentGamma(self.exchange) for _ in range(self.n_gamma_agents)]
        self.fund_price_series = array([])

    def create_fundamental_price_series(self, n_steps, random_seed: int):
        rand_state = RandomState(random_seed)
        evolution = rand_state.normal(size=n_steps)
        percentage_change = exp(
            evolution * self.fund_price_vol + self.fund_price_trend - 0.5 * self.fund_price_vol ** 2)
        time_series = percentage_change.cumprod() * self.initial_fund_price
        self.fund_price_series = time_series

    def clear_cache(self):
        super().clear_cache()
        self.fund_price_series = array([])

    def run(self, n_steps: int, average_trades_per_step: int, snapshot_interval: int,
            cancel_order_interval: int, random_seed: int = 42):
        self.clear_cache()

        rand_state = RandomState(random_seed)

        self.create_fundamental_price_series(n_steps, rand_state.randint(1000))

        n_draws = n_steps * average_trades_per_step * 2

        waiting_times = rand_state.exponential(1 / average_trades_per_step, size=n_draws)
        time_of_trade = waiting_times.cumsum()
        minute_of_trade = time_of_trade.astype(int)
        minute_of_trade = minute_of_trade[minute_of_trade < n_steps]
        n_trades = len(minute_of_trade)
        unique_minutes, count = np.unique(minute_of_trade, return_counts=True)
        max_trade_per_min = np.max(count)

        pr_fcn_agent = self.get_choice_between_fcn_and_gamma_traders(n_steps)
        choice_agent = rand_state.uniform(size=(n_steps, max_trade_per_min))
        choice_agent = (choice_agent < pr_fcn_agent).astype(int)

        fcn_agents = rand_state.randint(0, self.n_fcn_agents, size=n_trades)
        gamma_agents = rand_state.randint(0, self.n_gamma_agents, size=n_trades)
        agents_noise = rand_state.normal(0, 0.0001, size=n_trades)
        agents_volume = rand_state.randint(1, 5, size=n_trades)
        agents_buy_outcome = rand_state.uniform(size=n_trades)
        orders_to_cancel = []
        current_minute = 0
        changed_period = True
        minute_trade_counter = 0
        minutes_since_open = 0

        for i in range(n_trades):

            if current_minute != minute_of_trade[i]:
                changed_period = True
                current_minute = minute_of_trade[i]
                minutes_since_open = current_minute % MINUTES_IN_DAY
                minute_trade_counter = 0

            if current_minute % snapshot_interval == 0 and current_minute > 0 and changed_period:
                self.save_market_snapshot(current_minute)

            if changed_period:
                orders_to_cancel += [[]]

            if choice_agent[current_minute, minute_trade_counter] == 0:
                time_window = self.agents_fcn[fcn_agents[i]].submit_time_window()
                if time_window > current_minute - 1:
                    previous_price = None
                else:
                    if len(self.last_mid_price_series) >= time_window:
                        previous_price = self.last_mid_price_series[-time_window]
                    else:
                        previous_price = self.last_mid_price_series[0]

                self.agents_fcn[fcn_agents[i]].get_data(self.fund_price_series[current_minute], self.initial_fund_price,
                                                        previous_price, agents_noise[i])
                order_receipt = self.agents_fcn[fcn_agents[i]].decide_order(agents_volume[i], agents_buy_outcome[i])
                orders_to_cancel[-1] += [(fcn_agents[i], order_receipt.order_id)]
            else:
                previous_price = self.last_mid_price_series[-minutes_since_open]
                self.agents_gamma[gamma_agents[i]].get_data(self.initial_fund_price, previous_price)
                order_receipt = self.agents_gamma[gamma_agents[i]].decide_order()  # Currently undefined
                orders_to_cancel[-1] += [(fcn_agents[i], order_receipt.order_id)]

            if changed_period:
                self.mid_price_series.add(current_minute, self.exchange.mid_price)
                self.last_mid_price_series.add(current_minute, self.exchange.last_valid_mid_price)

            if current_minute >= cancel_order_interval and changed_period:
                for agent_id, order_id in orders_to_cancel[0]:
                    self.agents_fcn[agent_id].cancel_order(order_id)
                del orders_to_cancel[0]

            changed_period = False
            minute_trade_counter += 1

        self.market_snapshots.format_price_to_volume()

    @staticmethod
    def get_choice_between_fcn_and_gamma_traders(len_simulation):
        minutes_since_open = list(range(MINUTES_IN_DAY)) * (len_simulation // MINUTES_IN_DAY) \
                             + list(range(len_simulation % MINUTES_IN_DAY))
        assert len(minutes_since_open) == len_simulation
        minutes_since_open = np.array(minutes_since_open)
        weight_for_fcn = np.array([1] * len_simulation)
        tr = 50
        weight_for_gamma = np.exp(-(MINUTES_IN_DAY - tr - minutes_since_open) / tr)
        return weight_for_fcn / (weight_for_fcn + weight_for_gamma)
