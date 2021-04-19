from srvgd.plotting.cdf import plot_cdf

import pandas as pd
import matplotlib.pyplot as plt

data = {}
for key in ['no_logs', 'no_sin', 'only_logs', 'no_exp', 'all']:
    data[key] = pd.read_csv('01_rmse_{}.csv'.format(key))

for key in data:
    plot_cdf(data[key]['rmse_int'], label=key, ymax=1.)
plt.xlabel('RMSE in interpolation region')
plt.ylabel('Cummulative counts')
plt.legend()
plt.xscale('log')
plt.show()
