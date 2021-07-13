from market.simulator import SimulatorFCN, RandomSimulator
from market.exchange import Exchange
import matplotlib.pyplot as plt
import os
import time


def run_fnc_simulation(n_agents, n_steps, trades_per_step):
    exchange = Exchange()
    simulator_fcn = SimulatorFCN(exchange, n_agents, 500, 0.001)
    simulator_fcn.run(n_steps, trades_per_step, 5, 20)
    del exchange
    del simulator_fcn


def run_random_simulation(n_agents, n_steps, trades_per_step):
    exchange = Exchange()
    random_simulator = RandomSimulator(exchange, n_agents)
    random_simulator.run(n_steps, trades_per_step, 500, 5, 20)
    del exchange
    del random_simulator


if not os.path.exists('../results/visualize_simulation_time_complexity/'):
    os.makedirs('../results/visualize_simulation_time_complexity/')

n_agents_list = [100, 500, 1000, 5000, 10000]
n_agents_time_fcn = []
n_agents_time_random = []

n_steps_list = [100, 500, 1000, 5000, 10000]
n_steps_time_fcn = []
n_steps_time_random = []

trades_per_step_list = [1, 2, 5, 10, 20, 50]
trades_per_step_time_fcn = []
trades_per_step_time_random = []

for agents in n_agents_list:
    start = time.time()
    run_fnc_simulation(agents, 1000, 2)
    end = time.time()
    n_agents_time_fcn += [end - start]

    start = time.time()
    run_random_simulation(agents, 1000, 2)
    end = time.time()
    n_agents_time_random += [end - start]

for steps in n_steps_list:
    start = time.time()
    run_fnc_simulation(1000, steps, 2)
    end = time.time()
    n_steps_time_fcn += [end - start]

    start = time.time()
    run_random_simulation(1000, steps, 2)
    end = time.time()
    n_steps_time_random += [end - start]

for trades in trades_per_step_list:
    start = time.time()
    run_fnc_simulation(1000, 1000, trades)
    end = time.time()
    trades_per_step_time_fcn += [end - start]

    start = time.time()
    run_random_simulation(1000, 1000, trades)
    end = time.time()
    trades_per_step_time_random += [end - start]

plt.plot(n_agents_list, n_agents_time_random, label='random time')
plt.plot(n_agents_list, n_agents_time_fcn, label='fcn time')
plt.xlabel('n_agents')
plt.ylabel('time')
plt.title('Number of Agent vs Time')
plt.legend()
plt.savefig('../results/visualize_simulation_time_complexity/n_agents_vs_time.jpg')
plt.show()

plt.plot(n_steps_list, n_steps_time_random, label='random time')
plt.plot(n_steps_list, n_steps_time_fcn, label='fcn time')
plt.xlabel('n_steps')
plt.ylabel('time')
plt.title('Number of Steps vs Time')
plt.legend()
plt.savefig('../results/visualize_simulation_time_complexity/n_steps_vs_time.jpg')
plt.show()

plt.plot(trades_per_step_list, trades_per_step_time_random, label='random time')
plt.plot(trades_per_step_list, trades_per_step_time_fcn, label='fcn time')
plt.xlabel('trades_per_step')
plt.ylabel('time')
plt.title('Trades per Step vs Time')
plt.legend()
plt.savefig('../results/visualize_simulation_time_complexity/trades_per_step_vs_time.jpg')
plt.show()
