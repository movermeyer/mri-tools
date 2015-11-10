import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.stats import linregress
import matplotlib.patches as mpatches

from mri_tools.plots.layouts import SquareGridLayout
from mri_tools.plots.widgets import DiscreteSlider

__author__ = 'Robbert Harms'
__date__ = "2015-08-17"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ScatterDataInterface(object):

    def get_x_data(self, dimension=0):
        """Get the scatter data on the x axis

        Args:
            dimension (int): optional support for multi dimensional data

        Returns:
            ndarray: the data on the x axis
        """

    def get_y_data(self, dimension=0):
        """Get the scatter data on the y axis

        Args:
            dimension (int): optional support for multi dimensional data

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

    def get_nmr_dimensions(self):
        """Get the number of available dimensions.

        Returns:
            int: the available number of dimensions available
        """

class SimpleScatterData(ScatterDataInterface):

    def __init__(self, x_data, y_data, x_label, y_label, title):
        self._x_data = x_data
        self._y_data = y_data
        self._x_label = x_label
        self._y_label = y_label
        self._title = title

        if len(x_data.shape) != len(y_data.shape):
            raise ValueError('The x and y data should have the same number of dimensions.')

    def get_x_data(self, dimension=0):
        if self.get_nmr_dimensions() > 0:
            if dimension >= self._x_data.shape[1]:
                return self._x_data[:, -1]
            return self._x_data[:, dimension]
        return self._x_data

    def get_y_data(self, dimension=0):
        if self.get_nmr_dimensions() > 0:
            if dimension >= self._y_data.shape[1]:
                return self._y_data[:, -1]
            return self._y_data[:, dimension]
        return self._y_data

    def get_x_label(self):
        return self._x_label

    def get_y_label(self):
        return self._y_label

    def get_title(self):
        return self._title

    def get_nmr_dimensions(self):
        if len(self._x_data.shape) > 1:
            return self._x_data.shape[1]
        return 1


class ScatterDataInfo(object):

    def __init__(self, scatter_data_list, plot_titles):
        """All the information (including meta info) about the scatter data

        Args:
            scatter_data_list (list of ScatterDataInterface): the scatter data elements
            plot_titles (list of str): the titles of the dimensions
            nmr_dimensions (int): the default dimension
        """
        self._scatter_data_list = scatter_data_list
        self._plot_titles = plot_titles
        self._nmr_dimensions = max([sd.get_nmr_dimensions() for sd in scatter_data_list])

    def get_nmr_dimensions(self):
        """Get the number of supported dimensions.

        Returns:
            int: the number of dimensions
        """
        return self._nmr_dimensions

    def get_plot_title(self, dimension):
        """Get the title of this plot in the given dimension

        Args:
            dimension (int): the dimension from which we want the plot title

        Returns:
            str: the plot title
        """
        return self._plot_titles[dimension]

    def get_nmr_plots(self):
        """Get the number of plots we will display

        Returns:
            int: the number of plots
        """
        return len(self._scatter_data_list)

    def get_x_data(self, plot_ind, dimension=None):
        return self._scatter_data_list[plot_ind].get_x_data(dimension=dimension)

    def get_y_data(self, plot_ind, dimension=None):
        return self._scatter_data_list[plot_ind].get_y_data(dimension=dimension)

    def get_x_label(self, plot_ind):
        return self._scatter_data_list[plot_ind].get_x_label()

    def get_y_label(self, plot_ind):
        return self._scatter_data_list[plot_ind].get_y_label()

    def get_title(self, plot_ind):
        return self._scatter_data_list[plot_ind].get_title()


class ScatterPlots(object):

    def __init__(self, scatter_info, placement=None):
        """Create scatter plots of the given scatter data items.

        Args:
            scatter_info (ScatterDataInfo): the scatter data information
            placement (PlacementInterface): the placement options
        """
        self._scatter_info = scatter_info
        self.font_size = None
        self._figure = plt.figure(figsize=(18, 16))
        self.placement = placement or SquareGridLayout()
        self.show_titles = True
        self.dimension = 0
        self._dimension_slider = None
        self._updating_sliders = False
        self._show_sliders = True

    def show(self, dimension=0, show_titles=True, to_file=None, block=True, maximize=False, show_sliders=True):
        """Plot all the scatterplots.

        Args:
            dimension (int):
                The dimension to display
            show_titles (boolean): if we want to display the titles per scatter plot
            to_file (string, optional, default None):
                If to_file is not None it is supposed to be a filename where the image will be saved.
                If not set to None, nothing will be displayed, the results will directly be saved.
                Already existing items will be overwritten.
            block (boolean): If we want to block after calling the plots or not. Set this to False if you
                do not want the routine to block after drawing. In doing so you manually need to block.
            maximize (boolean): if we want to display the window maximized or not
            show_sliders (boolean): if we want to display the sliders
        """
        self.dimension = dimension
        self.show_titles = show_titles
        self._show_sliders = show_sliders

        self._setup()

        if maximize:
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()

        if to_file:
            plt.savefig(to_file)
            plt.close()
        else:
            plt.draw()
            if block:
                plt.show(True)

    def set_dimension(self, val):
        val = round(val)
        if not self._updating_sliders:
            self._updating_sliders = True
            self.dimension = int(round(val))

            if self.dimension > self._scatter_info.get_nmr_dimensions():
                self.dimension = self._scatter_info.get_nmr_dimensions()

            self._dimension_slider.set_val(val)
            self._rerender_maps()
            self._updating_sliders = False

    def _setup(self):
        if self.font_size:
            matplotlib.rcParams.update({'font.size': self.font_size})

        if self._show_sliders:
            ax = self._figure.add_axes([0.25, 0.008, 0.5, 0.01], axisbg='Wheat')
            self._dimension_slider = DiscreteSlider(
                ax, 'Volume', 0, self._scatter_info.get_nmr_dimensions() - 1,
                valinit=self.dimension, valfmt='%i', color='DarkSeaGreen', closedmin=True, closedmax=False)

            self._dimension_slider.on_changed(self.set_dimension)

        self._rerender_maps()

    def _rerender_maps(self):
        bb_min, bb_max = _get_bounding_box(self._scatter_info, self.dimension)

        for ind in range(self._scatter_info.get_nmr_plots()):
            axis = self.placement.get_axis(ind, self._scatter_info.get_nmr_plots())
            vf = axis.scatter(self._scatter_info.get_x_data(ind, dimension=self.dimension),
                              self._scatter_info.get_y_data(ind, dimension=self.dimension))

            axis.plot(np.arange(bb_min, bb_max, 0.01), np.arange(bb_min, bb_max, 0.01), 'k')

            slope, intercept, _, _, _ = linregress(self._scatter_info.get_x_data(ind, dimension=self.dimension),
                                                   self._scatter_info.get_y_data(ind, dimension=self.dimension))
            line_x_range = np.arange(bb_min, bb_max, 0.01)
            line_y_range = line_x_range * slope + intercept

            pos = np.where(np.logical_and(bb_min <= line_y_range, line_y_range<= bb_max))
            line_x_range = line_x_range[pos]
            line_y_range = line_y_range[pos]

            plt.plot(line_x_range, line_y_range, 'b')

            plt.xlabel(self._scatter_info.get_x_label(ind))
            plt.ylabel(self._scatter_info.get_y_label(ind))

            if self.show_titles:
                plt.title(self._scatter_info.get_plot_title(ind))

            plt.subplots_adjust(hspace=.3)

        self._figure.suptitle(self._scatter_info.get_plot_title(self.dimension))

        self._figure.canvas.draw()

        mng = plt.get_current_fig_manager()
        mng.canvas.set_window_title(self._scatter_info.get_plot_title(self.dimension))


class MultiROIScatterPlots(object):

    def __init__(self, scatter_info, placement=None):
        """Create scatter plots of the given scatter data items.

        Args:
            scatter_info (ScatterDataInfo): the scatter data information
            placement (PlacementInterface): the placement options
        """
        self._scatter_info = scatter_info
        self.font_size = None
        self._figure = plt.figure(figsize=(18, 16))
        self.placement = placement or SquareGridLayout()
        self.show_titles = True
        self.dimensions = []
        self._dimension_slider = None
        self._updating_sliders = False
        self._show_sliders = True
        self._legend_plot_options = {}

    def show(self, dimensions, show_titles=True, to_file=None, block=True, maximize=False, show_sliders=True,
             legend_plot_options=None):
        """Plot all the scatterplots with multiple dimensions per plot

        Args:
            dimensions (list of int):
                The dimensions to overlap on each other
            show_titles (boolean): if we want to display the titles per scatter plot
            to_file (string, optional, default None):
                If to_file is not None it is supposed to be a filename where the image will be saved.
                If not set to None, nothing will be displayed, the results will directly be saved.
                Already existing items will be overwritten.
            block (boolean): If we want to block after calling the plots or not. Set this to False if you
                do not want the routine to block after drawing. In doing so you manually need to block.
            maximize (boolean): if we want to display the window maximized or not
            show_sliders (boolean): if we want to display the sliders
            legend_plot_options (dict): the varargs for the legend, per plot
        """
        self.dimensions = dimensions
        self.show_titles = show_titles
        self._show_sliders = show_sliders
        self._legend_plot_options = legend_plot_options or {}
        self._setup()

        if maximize:
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()

        if to_file:
            plt.savefig(to_file)
            plt.close()
        else:
            plt.draw()
            if block:
                plt.show(True)

    def _rerender_maps(self):
        bb_min, bb_max = self._get_bounding_box()

        for ind in range(self._scatter_info.get_nmr_plots()):
            axis = self.placement.get_axis(ind, self._scatter_info.get_nmr_plots())
            axis.plot(np.arange(bb_min, bb_max, 0.01), np.arange(bb_min, bb_max, 0.01), 'k')

            color_cycle = axis._get_lines.color_cycle

            for dimension in self.dimensions:
                color = next(color_cycle)

                vf = axis.scatter(self._scatter_info.get_x_data(ind, dimension=dimension),
                                  self._scatter_info.get_y_data(ind, dimension=dimension), c=color)

                slope, intercept, _, _, _ = linregress(self._scatter_info.get_x_data(ind, dimension=dimension),
                                                       self._scatter_info.get_y_data(ind, dimension=dimension))

                line_x_range = np.arange(bb_min, bb_max, 0.01)
                line_y_range = line_x_range * slope + intercept

                pos = np.where(np.logical_and(bb_min <= line_y_range, line_y_range<= bb_max))
                line_x_range = line_x_range[pos]
                line_y_range = line_y_range[pos]

                plt.plot(line_x_range, line_y_range, color, label=self._scatter_info.get_plot_title(dimension))

            plt.xlabel(self._scatter_info.get_x_label(ind))
            plt.ylabel(self._scatter_info.get_y_label(ind))

            if self.show_titles:
                plt.title(self._scatter_info.get_plot_title(ind))

            plt.subplots_adjust(hspace=.3)

            legend_options = {}
            if ind in self._legend_plot_options:
                legend_options.update(self._legend_plot_options[ind])
            axis.legend(**legend_options)

        title = ', '.join(map(self._scatter_info.get_plot_title, self.dimensions))

        self._figure.suptitle(title)
        self._figure.canvas.draw()

        mng = plt.get_current_fig_manager()
        mng.canvas.set_window_title(title)

    def _get_bounding_box(self):
        min_list = []
        max_list = []
        for dimension in self.dimensions:
            bb_min, bb_max = _get_bounding_box(self._scatter_info, dimension)
            min_list.append(bb_min)
            max_list.append(bb_max)
        return max(min_list), min(max_list)

    def _setup(self):
        if self.font_size:
            matplotlib.rcParams.update({'font.size': self.font_size})
        self._rerender_maps()


def _get_bounding_box(scatter_info, dimension):
    """Get the bounding box for a scatter plot.

    Args:
        scatter_info (ScatterInfo): the information about the scatter plots
        dimension (int): which dimension we want to show
    """
    min_list = []
    max_list = []

    for ind in range(scatter_info.get_nmr_plots()):
        line_min = min(np.min(scatter_info.get_x_data(ind, dimension=dimension)),
                       np.min(scatter_info.get_y_data(ind, dimension=dimension)))
        line_max = max(np.max(scatter_info.get_x_data(ind, dimension=dimension)),
                       np.max(scatter_info.get_y_data(ind, dimension=dimension)))

        min_list.append(line_min)
        max_list.append(line_max)

    minimum = float(np.around(min(min_list), decimals=2))
    maximum = float(np.around(max(max_list), decimals=2))

    return max(minimum - 0.1, 0), min(maximum + 0.1, 1)
