from analysis.simulation_visualizer import VisualizeSimulationFCN
import matplotlib.pyplot as plt

if __name__ == "__main__":
    from market.simulator import SimulatorFCN
    from market.exchange import Exchange
    import os

    if not os.path.exists('../results/visualize_fcn/'):
        os.makedirs('../results/visualize_fcn/')

    exchange = Exchange()
    simulator_fcn = SimulatorFCN(exchange, 100, 500, 0.001, scale_fund=0.2, scale_chart=0.1, scale_noise=0.7)
    simulator_fcn.run(1000, 10, 5, 20)
    visualizer = VisualizeSimulationFCN(simulator_fcn)
    visualizer.plot_order_book()
    visualizer.plot_save_and_show("../results/visualize_fcn/fcn_lbo_heatmap.jpg")
    # visualizer.compute_agents_info()
    # visualizer.plot_agents_attributes_vs_parameters('value_portfolio')
    # visualizer.plot_save_and_show("../results/visualize_fcn/value_portfolio_vs_parameters.jpg", False)
    # visualizer.plot_agents_attributes_vs_parameters('cash')
    # visualizer.plot_save_and_show("../results/visualize_fcn/cash_vs_parameters.jpg", False)
    # visualizer.plot_agents_attributes_vs_parameters('stock')
    # visualizer.plot_save_and_show("../results/visualize_fcn/stock_vs_parameters.jpg", False)
    del exchange
    del simulator_fcn
    del visualizer

if __name__ == "__main__":
    from market.simulator import SimulatorFCN
    from market.exchange import Exchange

    exchange = Exchange()
    simulator_fcn = SimulatorFCN(exchange, 100, 500, 0.001, scale_fund=0.2, scale_chart=0.1, scale_noise=0.7)
    simulator_fcn.run(10000, 5, 5, 20)
    visualizer = VisualizeSimulationFCN(simulator_fcn)
    visualizer.plot_order_book()
    plt.vlines([3400, 4100, 4800, 5600], 360, 580, colors=['purple', 'yellow', 'brown', 'yellow'], linewidth=4)
    plt.ylim((360, 580))
    plt.grid(True)
    visualizer.plot_save_and_show("../results/visualize_fcn/fcn_lbo_heatmap_1000.jpg")
    # visualizer.compute_agents_info()
    # visualizer.plot_agents_attributes_vs_parameters('value_portfolio')
    # visualizer.plot_save_and_show("../results/visualize_fcn/value_portfolio_vs_parameters.jpg", False)
    # visualizer.plot_agents_attributes_vs_parameters('cash')
    # visualizer.plot_save_and_show("../results/visualize_fcn/cash_vs_parameters.jpg", False)
    # visualizer.plot_agents_attributes_vs_parameters('stock')
    # visualizer.plot_save_and_show("../results/visualize_fcn/stock_vs_parameters.jpg", False)
    del exchange
    del simulator_fcn
    del visualizer
