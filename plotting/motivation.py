
import numpy as np
import matplotlib.pyplot as plt

save_loc = '/Users/rgrindle/Dropbox/UVM/cs_masters/figures/'

x = np.linspace(-1, 1, 13)
f_pred = lambda x: np.exp(x)*np.sin(x)*np.cos(x)**2
f_true = lambda x: np.exp(x)

function_label = '\\hat{f}(x) = e^x\\sin(x)\\cos^2(x)$'

plt.figure()
plt.plot(x, f_true(x), 'o', label='dataset')
plt.xlabel('$x$')
plt.ylabel('$y$')
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend()
plt.savefig(save_loc+'motivation3.pdf')

more_x = np.linspace(np.min(x), np.max(x), 100)
plt.plot(more_x.flatten(), f_pred(more_x), '-', label=function_label)
plt.legend()
plt.savefig(save_loc+'motivation_with_function3.pdf')
plt.close('all')
