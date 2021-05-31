import pytest

from market.exchange import Exchange
from market.simulator import Simulator, RandomSimulator, SimulatorFCN


@pytest.fixture
def exchange():
    return Exchange()


@pytest.fixture
def simulator_100():
    exchange = Exchange()
    return Simulator(exchange, 100)


@pytest.fixture
def random_simulator_100():
    exchange = Exchange()
    return RandomSimulator(exchange, 100)


@pytest.fixture
def random_simulator_1000():
    exchange = Exchange()
    return RandomSimulator(exchange, 1000)


@pytest.fixture
def simulator_fcn_100():
    exchange = Exchange()
    return SimulatorFCN(exchange, 100, 500, 0.00001)
