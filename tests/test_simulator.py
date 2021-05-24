import pytest
from market.agent import Agent
from market.exchange import Exchange


def test_agent_subscribed_to_exchange():
    exchange = Exchange()
    agent = Agent(exchange)
    assert agent is exchange.subscribed_agents[agent.id]


def test_limit_order_submitted_to_exchange():
    assert False


def test_limit_order_cancelled_from_exchange():
    assert False


def test_market_order_is_submitted_and_executed():
    assert False


def test_limit_order_is_executed():
    assert False
