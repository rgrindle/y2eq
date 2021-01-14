import matplotlib.pyplot as plt
import pandas as pd

file_endname = '_layers10_clip1_dropoutTrue_lr1e-4_no_duplicates'
data = pd.read_csv('train_history{}.csv'.format(file_endname)).values
plt.plot(data[:, 0], label='train')
plt.plot(data[:, 1], label='val')
plt.xlabel('Epoch')
plt.ylabel('Symbolic loss')
plt.legend()
# plt.yscale('log')
plt.savefig('plot_history{}.pdf'.format(file_endname))
