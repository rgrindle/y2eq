"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Mar 9, 2021

PURPOSE: Plot a stacked bar graph where the shortest
         bars appear on top.

NOTES:

TODO:
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_bar_zordered(data):
    """data is a dictionary where the keys
    are the labels to be placed in the legend
    and each value is data for a bar graph in
    form of a dictionary (keys=xlabels, values=counts).
    """
    for _bin in data[list(data.keys())[0]]:
        prev_values = None
        for i, label in enumerate(data):
            if _bin not in data[label]:
                continue

            if i == 0:
                # Always make first bar have zorder=0.
                zorders = [0]
                # Initialize prev_values
                prev_values = [data[label][_bin]]
            else:
                # Use argsort on the
                # previous and current values
                # to determine the next zorder.
                indices = np.argsort(prev_values+[data[label][_bin]])

                # Get the index inside the argsorted indices
                # that corresponds to the current value
                index_index = np.where(indices == len(indices)-1)[0][0]

                if index_index == 0:
                    # Case 1: The current value is the
                    # smallest so far, so push to front
                    # with the largest zorder so far.
                    zorders.append(np.max(zorders)+1)
                elif index_index == len(indices)-1:
                    # Case 2: The current value is the
                    # largest so far, so push to the back
                    # with the smallest zorder so far.
                    zorders.append(np.min(zorders)-1)
                else:
                    # Case 3: The current value is neither
                    # the largest or smallest value so far
                    # so find the nearest zorders used so
                    # far and average them to get the zorder
                    # for this current value.
                    adjacent_z_less = zorders[indices[index_index-1]]
                    adjacent_z_greater = zorders[indices[index_index+1]]
                    zorders.append(np.mean([adjacent_z_less, adjacent_z_greater]))

                # Update prev_values
                prev_values.append(data[label][_bin])

            # Plot one stack (all at same x location)
            # of bars.
            plt.bar(_bin, data[label][_bin],
                    label=label, color='C{}'.format(i),
                    edgecolor='C{}'.format(i),
                    width=1.0,
                    zorder=zorders[-1])


if __name__ == '__main__':
    data = {'label1': {'a': 5, 'b': 10, 'c': 15},
            'label2': {'a': 7, 'b': 12, 'c': 5},
            'label3': {'a': 9, 'b': 2, 'c': 10}}

    plot_bar_zordered(data)
    plt.show()
