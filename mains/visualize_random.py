from analysis.simulation_visualizer import VisualizeSimulation

if __name__ == "__main__":
    from market.simulator import RandomSimulator
    from market.exchange import Exchange

    exchange = Exchange()
    random_simulator = RandomSimulator(exchange, 100)
    random_simulator.run(1000, 10, 500, 5, 20)
    visualizer = VisualizeSimulation(random_simulator)
    visualizer.plot_order_book()
    visualizer.plot_save_and_show("../results/random_lbo_heatmap.jpg")
    del exchange
    del random_simulator
    del visualizer

if __name__ == "__main__":
    from market.simulator import RandomSimulator
    from market.exchange import Exchange

    exchange = Exchange()
    random_simulator = RandomSimulator(exchange, 100)
    random_simulator.run(5000, 20, 500, 5, 20)
    visualizer = VisualizeSimulation(random_simulator)
    visualizer.plot_order_book()
    visualizer.plot_save_and_show("../results/random_lbo_heatmap_1000.jpg")
    del exchange
    del random_simulator
    del visualizer
