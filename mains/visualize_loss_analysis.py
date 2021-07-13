from itertools import product
import os

if not os.path.exists('../results/visualize_loss_analysis/'):
    os.mkdir('../results/visualize_loss_analysis/')

loss_analyzer = LossAnalyzer("../results/parameters_grid_search/grid_search_tests.csv")
f = loss_analyzer.results_df['scale_fund']
c = loss_analyzer.results_df['scale_chart']
n = loss_analyzer.results_df['scale_noise']
loss_analyzer.results_df['scale_fund'] = f / (f + c + n)
loss_analyzer.results_df['scale_chart'] = c / (f + c + n)
loss_analyzer.results_df['scale_noise'] = n / (f + c + n)

list_independent = ['scale_fund', 'scale_chart', 'scale_noise']
list_dependent = ['auto_correlation_loss', 'volatility_clustering_loss', 'leverage_effect_loss',
                  'distribution_loss', 'total_loss']

for (ind, dep) in product(list_independent, list_dependent):
    loss_analyzer.visualize_relationship(ind, dep, f'../results/visualize_loss_analysis/{ind}_vs_{dep}.jpg')
