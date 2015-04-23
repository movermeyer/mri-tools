import matplotlib.pyplot as plt
import numpy as np

__author__ = 'Robbert Harms'
__date__ = "2015-04-23"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class DVSPlot(object):

    def draw_table(self, table):
        """Enqueue a table plot.

        This will draw the table but will not block them. When run from the command line this will not display
        anything. In those cases call plot_block() when you are finished with the calculations.

        Args:
            dvs_table (DVSDirectionTable or ndarray): Either the DVS direction table object or a 3d array with the
                gradients.
        """
        fig, axes = plt.subplots(nrows=4, ncols=4)
        fig.tight_layout()

        ax = plt.subplot(211)
        plt.xticks(np.arange(0, table.shape[0]+1, 2.0))
        ax.set_title('gx, gy, gz')
        ax.plot(table[:, 0], 'r', label='gx')
        ax.plot(table[:, 1], 'b', label='gy')
        ax.plot(table[:, 2], 'g', label='gz')
        ax.legend(loc='upper right', prop={'size': 7})

        ax = plt.subplot(212)
        plt.xticks(np.arange(0, table.shape[0]+1, 2.0))
        ax.set_title('max(gx, gy, gz), abs diff(n, n+1)')
        ax.plot(np.max(table, axis=1), 'b', label='max')
        maxdiff = np.abs(np.diff(np.max(table, axis=1)))
        maxdiff = np.append(maxdiff, 0)
        ax.plot(maxdiff, 'g', label='abs diff')
        ax.legend(loc='upper right', prop={'size': 7})

        plt.draw()

    def plot_block(self):
        """Show all the drawn tables. This is a blocking call."""
        plt.show()