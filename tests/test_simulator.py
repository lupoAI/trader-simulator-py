import time

from market.simulator import RandomSimulator


def test_random_simulator_runs(exchange):
    random_simulator = RandomSimulator(exchange, 100)
    mid_price_time_series = random_simulator.run(1000, 10, 10000)
    assert len(mid_price_time_series) == 1000


def test_speed_simulator_1(exchange):
    start = time.time()
    random_simulator = RandomSimulator(exchange, 100)
    _ = random_simulator.run(1000, 10, 10000)
    end = time.time()
    simulation_runtime = end - start
    assert simulation_runtime < 0.5


def test_speed_simulator_2(exchange):
    start = time.time()
    random_simulator = RandomSimulator(exchange, 100)
    _ = random_simulator.run(1000, 100, 10000)
    end = time.time()
    simulation_runtime = end - start
    assert simulation_runtime < 5


def test_speed_simulator_3(exchange):
    start = time.time()
    random_simulator = RandomSimulator(exchange, 100)
    _ = random_simulator.run(10000, 10, 10000)
    end = time.time()
    simulation_runtime = end - start
    assert simulation_runtime < 5
