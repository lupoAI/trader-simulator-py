from market.agent import Agent
from market.data_model import Side


def test_agent_subscribed_to_exchange(exchange):
    agent = Agent(exchange)
    assert agent is exchange.agent_id_to_agent[agent.id]


def test_limit_orders_submitted_to_exchange(exchange):
    agent_buy = Agent(exchange)
    agent_sell = Agent(exchange)
    receipt_buy = agent_buy.limit_order(Side.BUY, 90, 10, True)
    receipt_sell = agent_sell.limit_order(Side.SELL, 100, 10, True)
    order_id_bid = receipt_buy.order_id
    order_id_ask = receipt_sell.order_id
    assertions = []
    assertions += [receipt_buy.outcome]
    assertions += [receipt_sell.outcome]
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
    assert assertions == [True] * 12


def test_limit_order_receipt(exchange):
    agent_buy = Agent(exchange)
    agent_sell = Agent(exchange)
    receipt_buy = agent_buy.limit_order(Side.BUY, 90, 10, True)
    receipt_sell = agent_sell.limit_order(Side.SELL, 100, 10, True)
    assertions = []
    assertions += [receipt_buy.side == Side.BUY]
    assertions += [receipt_buy.order_type == 'L']
    assertions += [receipt_buy.volume == 10]
    assertions += [receipt_buy.average_price == 90]
    assertions += [receipt_sell.side == Side.SELL]
    assertions += [receipt_sell.order_type == 'L']
    assertions += [receipt_sell.volume == 10]
    assertions += [receipt_sell.average_price == 100]
    assert assertions == [True] * 8


def test_limit_orders_are_remembered_by_agent(exchange):
    agent_buy = Agent(exchange)
    agent_sell = Agent(exchange)
    receipt_buy = agent_buy.limit_order(Side.BUY, 90, 10, True)
    receipt_sell = agent_sell.limit_order(Side.SELL, 100, 10, True)
    order_id_bid = receipt_buy.order_id
    order_id_ask = receipt_sell.order_id
    assertions = []
    assertions += [agent_buy.open_limit_orders[order_id_bid]
                   is exchange.order_id_to_order[order_id_bid]]
    assertions += [agent_sell.open_limit_orders[order_id_ask]
                   is exchange.order_id_to_order[order_id_ask]]
    assert assertions == [True] * 2


def test_market_order_fails_when_there_are_no_limit_orders(exchange):
    agent = Agent(exchange)
    order_receipt = agent.market_order(Side.BUY, 10, True)
    assertions = []
    assertions += [not order_receipt.outcome]
    assertions += [agent.cash == 0]
    assertions += [agent.stock == 0]
    assert assertions == [True] * 3


def test_market_order_partially_fulfills_when_not_enough_limit_orders(exchange):
    agent_limit = Agent(exchange)
    agent_market = Agent(exchange)
    limit_order_receipt = agent_limit.limit_order(Side.SELL, 100, 5, True)
    market_order_receipt = agent_market.market_order(Side.BUY, 10, True)
    assertions = []
    assertions += [limit_order_receipt.outcome]
    assertions += [market_order_receipt.outcome]
    assertions += [agent_limit.stock == -5]
    assertions += [agent_limit.cash == 500]
    assertions += [agent_limit.open_limit_orders == {}]
    assertions += [exchange.sorted_ask_prices == []]
    assertions += [agent_market.stock == 5]
    assertions += [agent_market.cash == -500]
    assertions += [market_order_receipt.average_price == 100]
    assertions += [market_order_receipt.volume == 5]
    assertions += [market_order_receipt.order_type == "M"]
    assert assertions == [True] * 11


def test_market_order_walks_the_book(exchange):
    agent_limit_1 = Agent(exchange)
    agent_limit_2 = Agent(exchange)
    agent_market = Agent(exchange)
    agent_limit_1.limit_order(Side.SELL, 100, 5)
    agent_limit_2.limit_order(Side.SELL, 110, 5)
    market_order_receipt = agent_market.market_order(Side.BUY, 10, True)
    assertions = []
    assertions += [agent_limit_1.open_limit_orders == {}]
    assertions += [agent_limit_2.open_limit_orders == {}]
    assertions += [exchange.sorted_ask_prices == []]
    assertions += [agent_limit_1.stock == -5]
    assertions += [agent_limit_2.stock == -5]
    assertions += [agent_limit_1.cash == 500]
    assertions += [agent_limit_2.cash == 550]
    assertions += [agent_market.stock == 10]
    assertions += [agent_market.cash == -1050]
    assertions += [market_order_receipt.outcome]
    assertions += [market_order_receipt.order_type == "M"]
    assertions += [market_order_receipt.volume == 10]
    assertions += [market_order_receipt.average_price == 105]
    assert assertions == [True] * 13


def test_market_order_walks_the_book_and_partially_fulfills(exchange):
    agent_limit_1 = Agent(exchange)
    agent_limit_2 = Agent(exchange)
    agent_market = Agent(exchange)
    agent_limit_1.limit_order(Side.SELL, 100, 5)
    limit_order_receipt = agent_limit_2.limit_order(Side.SELL, 110, 5, True)
    market_order_receipt = agent_market.market_order(Side.BUY, 8, True)
    limit_order_id = limit_order_receipt.order_id
    open_limit_order = agent_limit_2.open_limit_orders[limit_order_id]
    assertions = []
    assertions += [agent_limit_1.open_limit_orders == {}]
    assertions += [exchange.sorted_ask_prices == [110]]
    assertions += [exchange.order_id_to_order[limit_order_id]
                   is open_limit_order]
    assertions += [open_limit_order.volume == 2]
    assertions += [market_order_receipt.average_price == 830 / 8]
    assertions += [agent_limit_2.cash == 330]
    assertions += [agent_limit_2.stock == -3]
    assertions += [agent_market.cash == -830]
    assertions += [agent_market.stock == 8]
    assert assertions == [True] * 9


def test_market_order_fulfills_multiple_limit_orders(exchange):
    agent_limit_1 = Agent(exchange)
    agent_limit_2 = Agent(exchange)
    agent_limit_3 = Agent(exchange)
    agent_market = Agent(exchange)
    agent_limit_1.limit_order(Side.SELL, 100, 10)
    agent_limit_2.limit_order(Side.SELL, 100, 10)
    agent_limit_3.limit_order(Side.SELL, 100, 10)
    market_order_receipt = agent_market.market_order(Side.BUY, 30, True)
    assertions = []
    assertions += [agent_limit_1.open_limit_orders == {}]
    assertions += [agent_limit_2.open_limit_orders == {}]
    assertions += [agent_limit_3.open_limit_orders == {}]
    assertions += [exchange.order_id_to_order == {}]
    assertions += [agent_limit_1.cash == 1000]
    assertions += [agent_limit_2.cash == 1000]
    assertions += [agent_limit_3.cash == 1000]
    assertions += [agent_limit_1.stock == -10]
    assertions += [agent_limit_2.stock == -10]
    assertions += [agent_limit_3.stock == -10]
    assertions += [agent_market.stock == 30]
    assertions += [agent_market.cash == -3000]
    assertions += [market_order_receipt.outcome]
    assertions += [market_order_receipt.order_type == "M"]
    assertions += [market_order_receipt.volume == 30]
    assertions += [market_order_receipt.average_price == 100]
    assert assertions == [True] * 16


def test_market_order_partially_fulfills_limit_order(exchange):
    agent_limit = Agent(exchange)
    agent_market = Agent(exchange)
    limit_order_receipt = agent_limit.limit_order(Side.SELL, 100, 15, True)
    agent_market.market_order(Side.BUY, 10)
    limit_order_id = limit_order_receipt.order_id
    limit_order = agent_limit.open_limit_orders[limit_order_id]
    assertions = []
    assertions += [exchange.order_id_to_order[limit_order_id]
                   is limit_order]
    assertions += [limit_order.volume == 5]
    assertions += [agent_limit.cash == 1000]
    assertions += [agent_limit.stock == -10]
    assertions += [agent_market.cash == -1000]
    assertions += [agent_market.stock == 10]
    assert assertions == [True] * 6


def test_market_order_partially_fulfills_second_limit_order(exchange):
    agent_limit_1 = Agent(exchange)
    agent_limit_2 = Agent(exchange)
    agent_market = Agent(exchange)
    agent_limit_1.limit_order(Side.SELL, 100, 5)
    limit_order_receipt = agent_limit_2.limit_order(Side.SELL, 100, 5, True)
    market_order_receipt = agent_market.market_order(Side.BUY, 8, True)
    limit_order_id = limit_order_receipt.order_id
    open_limit_order = agent_limit_2.open_limit_orders[limit_order_id]
    assertions = []
    assertions += [agent_limit_1.open_limit_orders == {}]
    assertions += [exchange.sorted_ask_prices == [100]]
    assertions += [exchange.order_id_to_order[limit_order_id]
                   is open_limit_order]
    assertions += [open_limit_order.volume == 2]
    assertions += [market_order_receipt.average_price == 800 / 8]
    assertions += [agent_limit_2.cash == 300]
    assertions += [agent_limit_2.stock == -3]
    assertions += [agent_market.cash == -800]
    assertions += [agent_market.stock == 8]
    assert assertions == [True] * 9


def test_limit_order_is_executed(exchange):
    agent_buy = Agent(exchange)
    agent_sell = Agent(exchange)

    receipt_limit_buy = agent_buy.limit_order(Side.BUY, 100, 10, True)
    agent_sell.limit_order(Side.SELL, 90, 4)

    limit_buy_id = receipt_limit_buy.order_id
    limit_buy_order = agent_buy.open_limit_orders[limit_buy_id]
    assert limit_buy_order.volume == 6


def test_best_price_changes(exchange):
    agent_limit = Agent(exchange)
    agent_market = Agent(exchange)
    agent_limit.limit_order(Side.BUY, 100, 10)
    agent_limit.limit_order(Side.BUY, 101, 10)
    assertions = []
    assertions += [exchange.best_bid_price == 101]
    agent_market.market_order(Side.SELL, 15)
    assertions += [exchange.best_bid_price == 100]
    assert assertions == [True] * 2


def test_limit_order_cancelled_from_exchange(exchange):
    agent_buy = Agent(exchange)
    agent_sell = Agent(exchange)
    agent_buy.limit_order(Side.BUY, 100, 10)
    buy_receipt = agent_buy.limit_order(Side.BUY, 100, 10, True)
    agent_buy.limit_order(Side.BUY, 90, 10)
    agent_buy.limit_order(Side.BUY, 80, 10)
    sell_receipt = agent_sell.limit_order(Side.SELL, 110, 10, True)
    agent_sell.limit_order(Side.SELL, 120, 10)
    agent_sell.limit_order(Side.SELL, 120, 10)
    agent_sell.limit_order(Side.SELL, 130, 10)
    agent_buy.cancel_order(buy_receipt.order_id)
    agent_sell.cancel_order(sell_receipt.order_id)
    assertions = []
    assertions += [exchange.best_ask_price == 120]
    assertions += [buy_receipt.order_id not in agent_buy.open_limit_orders]
    assertions += [sell_receipt.order_id not in agent_sell.open_limit_orders]
    assertions += [buy_receipt.order_id not in exchange.order_id_to_order]
    assertions += [buy_receipt.order_id not in exchange.order_id_to_agent_id]
    assertions += [110 not in exchange.sorted_ask_prices]
    assertions += [len(exchange.price_to_orders_ids[100]) == 1]
    assert assertions == [True] * 7


