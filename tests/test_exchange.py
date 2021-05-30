def test_exchange_can_find_agent_from_id(exchange):
    assert exchange.agent_id_to_agent == {}


def test_exchange_can_find_order_from_id(exchange):
    assert exchange.order_id_to_order == {}


def test_exchange_can_map_an_order_to_agent(exchange):
    assert exchange.order_id_to_agent_id == {}


def test_exchange_can_map_prices_to_limit_orders(exchange):
    assert exchange.price_to_orders_ids == {}


def test_exchange_can_remember_available_bid_prices(exchange):
    assert exchange.sorted_bid_prices == []


def test_exchange_can_remember_available_ask_prices(exchange):
    assert exchange.sorted_ask_prices == []


def test_exchange_can_track_best_bid_price(exchange):
    assert exchange.best_bid_price is None


def test_exchange_can_track_best_ask_price(exchange):
    assert exchange.best_ask_price is None


def test_exchange_can_track_best_bid_volume(exchange):
    assert exchange.best_bid_volume is None


def test_exchange_can_track_best_ask_volume(exchange):
    assert exchange.best_ask_volume is None


def test_exchange_has_a_mid_price(exchange):
    assert exchange.mid_price is None


def test_exchange_has_a_last_valid_mid_price(exchange):
    assert exchange.last_valid_mid_price is None


def test_exchange_has_total_bid_volume(exchange):
    assert exchange.total_bid_volume == 0


def test_exchange_has_total_ask_volume(exchange):
    assert exchange.total_ask_volume == 0


def test_exchange_can_track_price_to_volume(exchange):
    assert exchange.price_to_volume == {}
