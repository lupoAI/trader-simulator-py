from market.agent import Agent


def test_agent_has_exchange(exchange):
    agent = Agent(exchange)
    assert agent.exchange is exchange


def test_agent_is_subscribed_to_exchange(exchange):
    agent = Agent(exchange)
    assert agent is agent.exchange.agent_id_to_agent[agent.id]


def test_agent_has_structure_to_store_orders(exchange):
    agent = Agent(exchange)
    assert agent.open_limit_orders == {}


def test_agent_has_cash_variable(exchange):
    agent = Agent(exchange)
    assert agent.cash == 0


def test_agent_has_stock_variable(exchange):
    agent = Agent(exchange)
    assert agent.stock == 0
