import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file_endname_list = ['_layers10_clip1_dropoutFalse_lr1e-4_1100',
                     '_layers10_clip1_dropoutFalse_lr1e-4_1400',
                     '_layers10_clip1_dropoutFalse_lr1e-4_1700']

epoch_count = 0
checkpoints = []
for file_endname in file_endname_list:
    data = pd.read_csv('train_history{}.csv'.format(file_endname)).values
    i = np.argmin(data[:, 1])
    epoch_list = list(range(epoch_count, epoch_count+i+1))
    plt.plot(epoch_list, data[:i+1, 0], label='train', color='C0')
    plt.plot(epoch_list, data[:i+1, 1], label='val', color='C1')
    checkpoints.append(epoch_list[-1])
    epoch_count += i+1

# Do draw checkpoint lines after to
# avoid different length lines.
plt.vlines(checkpoints[:-1], *plt.ylim(), 'k', label='checkpoint',
           linestyles='dashed')

plt.xlabel('Epoch')
plt.ylabel('Symbolic loss')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
# plt.yscale('log')
plt.title('Figure 1: Loss decreases, but slowly')
plt.savefig('plot_history2{}.pdf'.format(file_endname))
