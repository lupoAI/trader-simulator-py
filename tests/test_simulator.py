from market.simulator import RandomSimulator


def test_random_simulator_runs(exchange):
    random_simulator = RandomSimulator(exchange, 100)
    mid_price_time_series = random_simulator.run(1000, 10, 10000)
    assert len(mid_price_time_series) == 1000
