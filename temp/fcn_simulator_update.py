from numpy import exp, array
from numpy.random import RandomState

from market.agent import AgentFCN
from market.exchange import Exchange

from market.simulator import Simulator


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
