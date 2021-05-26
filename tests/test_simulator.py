from market.agent import Agent
from market.data_model import Side
from numpy import inf


def test_exchange_has_all_expected_elements(exchange):
    assertions = []
    assertions += [exchange.agent_id_to_agent == {}]
    assertions += [exchange.order_id_to_order == {}]
    assertions += [exchange.order_id_to_agent_id == {}]
    assertions += [exchange.price_to_orders_ids == {}]
    assertions += [exchange.sorted_bid_prices == []]
    assertions += [exchange.sorted_ask_prices == []]
    assertions += [exchange.best_bid_price == 0]
    assertions += [exchange.best_ask_price == inf]
    assertions += [exchange.best_bid_volume == 0]
    assertions += [exchange.best_ask_volume == 0]
    assertions += [exchange.true_price == 10000]
    assert assertions == [True] * 11


def test_agent_has_all_expected_elements(exchange):
    agent = Agent(exchange)
    assertions = []
    assertions += [agent.exchange is exchange]
    assertions += [agent is agent.exchange.agent_id_to_agent[agent.id]]
    assertions += [agent.open_limit_orders == {}]
    assert assertions == [True] * 3


def test_agent_subscribed_to_exchange(exchange):
    agent = Agent(exchange)
    assert agent is exchange.agent_id_to_agent[agent.id]


def test_limit_orders_submitted_to_exchange(exchange):
    agent = Agent(exchange)
    order_id_bid = agent.limit_order(Side.BUY, 90, 10)
    order_id_ask = agent.limit_order(Side.SELL, 100, 10)
    assertions = []
    assertions += [order_id_bid in exchange.order_id_to_agent_id]
    assertions += [order_id_ask in exchange.order_id_to_agent_id]
    assertions += [order_id_bid in exchange.order_id_to_order]
    assertions += [order_id_ask in exchange.order_id_to_order]
    assertions += [order_id_bid in exchange.price_to_orders_ids[90]]
    assertions += [order_id_ask in exchange.price_to_orders_ids[100]]
    assertions += [[90] == exchange.sorted_bid_prices]
    assertions += [[100] == exchange.sorted_ask_prices]
    assertions += [90 == exchange.best_bid_price]
    assertions += [100 == exchange.best_ask_price]
    assert assertions == [True] * 10


def test_market_order_is_submitted_and_executed(exchange):
    # TODO
    assert False


def test_limit_order_cancelled_from_exchange(exchange):
    # TODO
    assert False


def test_best_price_changes(exchange):
    agent1 = Agent(exchange)
    agent2 = Agent(exchange)

    _ = agent1.limit_order(Side.BUY, 100, 10)
    _ = agent1.limit_order(Side.BUY, 101, 10)
    assert exchange.best_bid_price == 101
    _ = agent2.market_order(Side.SELL, 15)
    assert exchange.best_bid_price == 100
    # TODO


def test_walking_the_book_with_non_adjacent_ticks(exchange):
    agent1 = Agent(exchange)
    agent2 = Agent(exchange)

    _ = agent1.limit_order(Side.BUY, 100, 10)
    _ = agent1.limit_order(Side.BUY, 1000, 10)
    assert exchange.best_bid_price == 1000
    _ = agent2.market_order(Side.SELL, 15)
    assert exchange.best_bid_price == 100
    # TODO


def test_limit_order_is_executed(exchange):
    agent1 = Agent(exchange)
    agent2 = Agent(exchange)

    order_id1 = agent1.limit_order(Side.BUY, 100, 10)
    _ = agent2.limit_order(Side.SELL, 90, 4)

    order1 = exchange.order_id_to_order[order_id1]
    assert order1.volume == 6
    # TODO
