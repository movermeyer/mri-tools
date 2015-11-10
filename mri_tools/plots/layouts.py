import itertools
import math
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

__author__ = 'Robbert Harms'
__date__ = "2015-11-06"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class GridLayout(object):

    def __init__(self):
        self.spacings = dict(left=0.04, right=0.96, top=0.95, bottom=0.07)

    def get_axis(self, index, nmr_plots):
        """Get the axis for the subplot at the given index in the data list.

        Args:
            index (int): the index of the subplot in the list of plots
            nmr_plots (int): the total number of plots

        Returns:
            axis: a matplotlib axis object that can be drawn on
        """


class SquareGridLayout(GridLayout):

    def get_axis(self, index, nmr_plots):
        rows, cols = self._get_row_cols_square(nmr_plots)
        grid = GridSpec(rows, cols, **self.spacings)
        return plt.subplot(grid[index])

    def _get_row_cols_square(self, nmr_plots):
        defaults = ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (2, 3), (2, 3))
        if nmr_plots < len(defaults):
            return defaults[nmr_plots - 1]
        else:
            cols = math.ceil(nmr_plots / 3.0)
            rows = math.ceil(float(nmr_plots) / cols)
            rows = int(rows)
            cols = int(cols)
        return rows, cols


class LowerTriangleGridLayout(GridLayout):

    def __init__(self, size):
        super(LowerTriangleGridLayout, self).__init__()
        self._size = size
        self._positions = []

        for y, x in itertools.product(range(self._size), range(self._size)):
            if x >= y:
                self._positions.append(x * self._size + y)

    def get_axis(self, index, nmr_plots):
        grid = GridSpec(self._size, self._size, **self.spacings)
        return plt.subplot(grid[self._positions[index]])


class SingleColumnGridLayout(GridLayout):

    def get_axis(self, index, nmr_plots):
        grid = GridSpec(nmr_plots, 1, **self.spacings)
        return plt.subplot(grid[index])


class SingleRowGridLayout(GridLayout):

    def get_axis(self, index, nmr_plots):
        grid = GridSpec(1, nmr_plots, **self.spacings)
        return plt.subplot(grid[index])
