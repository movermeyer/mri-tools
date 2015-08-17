import math
import itertools
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.stats import linregress

__author__ = 'Robbert Harms'
__date__ = "2015-08-17"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ScatterDataInterface(object):

    def get_x_data(self):
        """Get the scatter data on the x axis

        Returns:
            ndarray: the data on the x axis
        """

    def get_y_data(self):
        """Get the scatter data on the y axis

        Returns:
            ndarray: the data on the y axis
        """

    def get_x_label(self):
        """Get the label on the x axis

        Returns:
            str: the label on the x axis
        """

    def get_y_label(self):
        """Get the label on the y axis

        Returns:
            str: the label on the y axis
        """

    def get_title(self):
        """Get the title of this scatter data

        Returns:
            str: the title of this scatter data
        """


class SimpleScatterData(ScatterDataInterface):

    def __init__(self, x_data, y_data, x_label, y_label, title):
        self._x_data = x_data
        self._y_data = y_data
        self._x_label = x_label
        self._y_label = y_label
        self._title = title

    def get_x_data(self):
        return self._x_data

    def get_y_data(self):
        return self._y_data

    def get_x_label(self):
        return self._x_label

    def get_y_label(self):
        return self._y_label

    def get_title(self):
        return self._title


class PlacementInterface(object):

    def get_axis(self, index, nmr_plots):
        """Get the axis for the subplot at the given index in the data list.

        Args:
            index (int): the index of the subplot in the list of plots
            nmr_plots (int): the total number of plots

        Returns:
            axis: a matplotlib axis object that can be drawn on
        """


class SquarePlacement(PlacementInterface):

    def get_axis(self, index, nmr_plots):
        rows, cols = self._get_row_cols_square(nmr_plots)
        grid = GridSpec(rows, cols, left=0.04, right=0.96, top=0.97, bottom=0.07)
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


class LowerTrianglePlacement(PlacementInterface):

    def __init__(self, size):
        self._size = size
        self._positions = []

        for y, x in itertools.product(range(self._size), range(self._size)):
            if x >= y:
                self._positions.append(x * self._size + y)

    def get_axis(self, index, nmr_plots):
        grid = GridSpec(self._size, self._size, left=0.04, right=0.96, top=0.97, bottom=0.07)
        return plt.subplot(grid[self._positions[index]])


class ScatterPlots(object):

    def __init__(self, scatter_data_list, placement=None):
        """Create scatter plots of the given scatter data items.

        Args:
            scatter_data_list (list of ScatterDataInterface): the scatter data elements
            placement (PlacementInterface): the placement options
        """
        self._scatter_data_list = scatter_data_list

        self.font_size = None
        self._figure = plt.figure(figsize=(18, 16))
        self.placement = placement or SquarePlacement()
        self.display_titles = True
        self.suptitle = None

    def show(self, display_titles=True, to_file=None, block=True, maximize=False, window_title=None, suptitle=None):
        self.display_titles = display_titles
        self.suptitle = suptitle

        self._setup()

        if maximize:
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()

        if window_title:
            mng = plt.get_current_fig_manager()
            mng.canvas.set_window_title(window_title)

        if to_file:
            plt.savefig(to_file)
            plt.close()
        else:
            plt.draw()
            if block:
                plt.show(True)

    def _setup(self):
        if self.font_size:
            matplotlib.rcParams.update({'font.size': self.font_size})

        self._rerender_maps()

    def _rerender_maps(self):
        for ind, scatter_data in enumerate(self._scatter_data_list):
            slope, intercept, _, _, _ = linregress(scatter_data.get_x_data(), scatter_data.get_y_data())

            axis = self.placement.get_axis(ind, len(self._scatter_data_list))
            vf = axis.scatter(scatter_data.get_x_data(), scatter_data.get_y_data())

            line_min, line_max = self._get_bounding_box()

            axis.plot(np.arange(line_min, line_max, 0.1), np.arange(line_min, line_max, 0.1), 'g')

            line_x_range = np.arange(line_min, line_max, 0.1)
            line_y_range = line_x_range * slope + intercept
            plt.plot(line_x_range, line_y_range, 'r')

            plt.xlabel(scatter_data.get_x_label())
            plt.ylabel(scatter_data.get_y_label())

            if self.display_titles:
                plt.title(scatter_data.get_title())

            plt.subplots_adjust(hspace=.3)

        self._figure.canvas.draw()

        if self.suptitle:
            self._figure.suptitle(self.suptitle)

    def _get_bounding_box(self):
        min_list = []
        max_list = []

        for scatter_data in self._scatter_data_list:
            line_min = min(np.min(scatter_data.get_x_data()), np.min(scatter_data.get_y_data()))
            line_max = max(np.max(scatter_data.get_x_data()), np.max(scatter_data.get_y_data()))

            min_list.append(line_min)
            max_list.append(line_max)

        min_x = min(min_list)
        max_x = max(max_list)
        min_list = []
        max_list = []

        for scatter_data in self._scatter_data_list:
            slope, intercept, _, _, _ = linregress(scatter_data.get_x_data(), scatter_data.get_y_data())
            min_list.append(min_x * slope + intercept)
            max_list.append(max_x * slope + intercept)

        return min(min_list), max(max_list)