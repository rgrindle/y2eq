"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jul 13, 2021

PURPOSE: Show an example of numerically similar
         functional forms.

NOTES:

TODO:
"""
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12.8, 4.8))
plt.subplots_adjust(left=0.05, right=0.5, top=0.95, bottom=0.1)
ax = fig.add_subplot(111)
ax.set_ylim([-1.5, 11])
x = np.linspace(-4, 6, 1000)
f_i = lambda x: 0.1*np.exp(x)
f_j = lambda x: -2.8533396*np.sin(0.46994925*x + 1.31815196) + 3.

# These are in unitless percentages of the figure size. (0,0 is bottom left)
left, bottom, width, height = [0.6, 0.1, 0.3, 0.8]
inset_ax = fig.add_axes([left, bottom, width, height])
inset_ax.set_xlim([0, 3.1])
inset_ax.set_ylim([0, 2.5])
mark_inset(ax, inset_ax, loc1=3,
           loc2=2, fc='none', ec='0.5')

inset_ax.plot(x, f_i(x), 'C3')
inset_ax.plot(x, f_j(x), 'C0')

inset_ax.set_xlabel('$x$', fontsize=12)
inset_ax.set_ylabel('$y$', fontsize=12)
inset_ax.set_xticks(range(0, 4))
inset_ax.set_xticklabels(range(0, 4), fontsize=12)
inset_ax.set_yticklabels([0.5*i for i in range(6)], fontsize=12)

ax.plot(x, f_i(x), 'C3',
        label='$f_1(x; \\vec{\\theta}_1) = 0.1e^x$')
ax.plot(x, f_j(x), 'C0',
        label='$f_2(x; \\vec{\\theta}_2) = -2.85\\sin(0.47x + 1.32) + 3$')

ax.set_xlabel('$x$', fontsize=12)
ax.set_ylabel('$y$', fontsize=12)
ax.set_xticklabels(range(-6, 7, 2), fontsize=12)
ax.set_yticklabels(range(-2, 11, 2), fontsize=12)

ax.legend(fontsize=12)
plt.savefig('numerically_similar_ff_examples.pdf')
