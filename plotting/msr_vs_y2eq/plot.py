from srvgd.plotting.cdf import plot_cdf

import matplotlib.pyplot as plt
import pandas as pd

msr_data = pd.read_csv('mrs_data.csv', header=None).values.flatten()
y2eq_data = pd.read_csv('y2eq_data.csv', header=None).values.flatten()
print(msr_data)
print(y2eq_data)
plot_cdf(msr_data, labels=False, label='TLC-SR')
plot_cdf(y2eq_data, labels=False, label='y2eq')
plt.xlabel('RMSE')
plt.ylabel('Cummulative counts')
# plt.title('train data')
plt.legend()
plt.show()
