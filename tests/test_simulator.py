import time


def test_simulator_has_exchange(simulator_100):
    assert simulator_100.exchange is not None


def test_simulator_has_number_of_agents(simulator_100):
    assert simulator_100.n_agents == 100


def test_random_simulator_runs(random_simulator_100):
    random_simulator_100.run(1000, 10, 10000, 20, 60)
    assert len(random_simulator_100.mid_price_series.price) == 1000


def test_simulator_fcn_runs(simulator_fcn_100):
    simulator_fcn_100.run(1000, 10, 20, 60)
    assert len(simulator_fcn_100.mid_price_series.price) == 1000


def test_speed_random_simulator_1(random_simulator_100):
    start = time.time()
    random_simulator_100.run(1000, 10, 10000, 20, 60)
    end = time.time()
    simulation_runtime = end - start
    assert simulation_runtime < 0.6


def test_speed_random_simulator_2(random_simulator_100):
    start = time.time()
    random_simulator_100.run(1000, 100, 10000, 20, 60)
    end = time.time()
    simulation_runtime = end - start
    assert simulation_runtime < 6


def test_speed_random_simulator_3(random_simulator_100):
    start = time.time()
    random_simulator_100.run(10000, 10, 10000, 20, 60)
    end = time.time()
    simulation_runtime = end - start
    assert simulation_runtime < 6


def test_speed_fcn_simulator_1(simulator_fcn_100):
    start = time.time()
    simulator_fcn_100.run(1000, 10, 20, 60)
    end = time.time()
    simulation_runtime = end - start
    assert simulation_runtime < 0.6


def test_speed_fcn_simulator_2(simulator_fcn_100):
    start = time.time()
    simulator_fcn_100.run(1000, 100, 20, 60)
    end = time.time()
    simulation_runtime = end - start
    assert simulation_runtime < 6
