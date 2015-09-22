import math
import itertools
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import Slider
import numpy as np
from scipy.stats import linregress

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

    def get_y_data(self,dimension=0):
        """Get the scatter data on the y axis

        Args:
            dimension (int): optional support for multi dimensional data

        Returns:
            ndarray: the data on the y axis
        """

    def get_x_label(self,dimension=0):
        """Get the label on the x axis

        Args:
            dimension (int): optional support for multi dimensional data

        Returns:
            str: the label on the x axis
        """

    def get_y_label(self,dimension=0):
        """Get the label on the y axis

        Args:
            dimension (int): optional support for multi dimensional data

        Returns:
            str: the label on the y axis
        """

    def get_title(self,dimension=0):
        """Get the title of this scatter data

        Args:
            dimension (int): optional support for multi dimensional data

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

    def get_x_data(self, dimension=0):
        return self._x_data

    def get_y_data(self, dimension=0):
        return self._y_data

    def get_x_label(self, dimension=0):
        return self._x_label

    def get_y_label(self, dimension=0):
        return self._y_label

    def get_title(self, dimension=0):
        return self._title


class MultiDimensionalScatterData(ScatterDataInterface):

    def __init__(self, x_data, y_data, x_label, y_label, title):
        self._x_data = x_data
        self._y_data = y_data
        self._x_label = x_label
        self._y_label = y_label
        self._title = title

    def get_x_data(self, dimension=0):
        return self._x_data[:, dimension]

    def get_y_data(self, dimension=0):
        return self._y_data[:, dimension]

    def get_x_label(self, dimension=0):
        return self._x_label

    def get_y_label(self, dimension=0):
        return self._y_label

    def get_title(self, dimension=0):
        return self._title


class ScatterDataInfo(object):

    def __init__(self, scatter_data_list, plot_titles, default_dimension, nmr_dimensions):
        """All the information (including meta info) about the scatter data

        Args:
            scatter_data_list (list of ScatterDataInterface): the scatter data elements
            plot_titles (list of str): the titles of the dimensions
            default_dimension (int): the default dimension
        """
        self._scatter_data_list = scatter_data_list
        self._plot_titles = plot_titles
        self._default_dimension = default_dimension
        self._nmr_dimensions = nmr_dimensions

    def get_nmr_dimensions(self):
        """Get the number of supported dimensions.

        Returns:
            int: the number of dimensions
        """
        return self._nmr_dimensions

    def get_plot_title(self, dimension=None):
        """Get the title of this plot in the given dimension

        Args:
            dimension (int): the dimension from which we want the plot title

        Returns:
            str: the plot title
        """
        dimension = dimension or self._default_dimension
        return self._plot_titles[dimension]

    def get_nmr_plots(self):
        """Get the number of plots we will display

        Returns:
            int: the number of plots
        """
        return len(self._scatter_data_list)

    def get_x_data(self, plot_ind, dimension=None):
        dimension = dimension or self._default_dimension
        return self._scatter_data_list[plot_ind].get_x_data(dimension=dimension)

    def get_y_data(self, plot_ind, dimension=None):
        dimension = dimension or self._default_dimension
        return self._scatter_data_list[plot_ind].get_y_data(dimension=dimension)

    def get_x_label(self, plot_ind, dimension=0):
        dimension = dimension or self._default_dimension
        return self._scatter_data_list[plot_ind].get_x_label(dimension=dimension)

    def get_y_label(self, plot_ind, dimension=0):
        dimension = dimension or self._default_dimension
        return self._scatter_data_list[plot_ind].get_y_label(dimension=dimension)

    def get_title(self, plot_ind, dimension=0):
        dimension = dimension or self._default_dimension
        return self._scatter_data_list[plot_ind].get_title(dimension=dimension)

#todo use GridLayout from MDT.visualization
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

    def __init__(self, scatter_info, placement=None):
        """Create scatter plots of the given scatter data items.

        Args:
            scatter_info (ScatterDataInfo): the scatter data information
            placement (PlacementInterface): the placement options
        """
        self._scatter_info = scatter_info
        self.font_size = None
        self._figure = plt.figure(figsize=(18, 16))
        self.placement = placement or SquarePlacement()
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
            self._dimension_slider = _DiscreteSlider(
                ax, 'Volume', 0, self._scatter_info.get_nmr_dimensions() - 1,
                valinit=self.dimension, valfmt='%i', color='DarkSeaGreen', closedmin=True, closedmax=False)

            self._dimension_slider.on_changed(self.set_dimension)

        self._rerender_maps()

    def _rerender_maps(self):
        bb_min, bb_max = self._get_bounding_box()

        for ind in range(self._scatter_info.get_nmr_plots()):
            axis = self.placement.get_axis(ind, self._scatter_info.get_nmr_plots())
            vf = axis.scatter(self._scatter_info.get_x_data(ind, dimension=self.dimension),
                              self._scatter_info.get_y_data(ind, dimension=self.dimension))

            axis.plot(np.arange(bb_min, bb_max, 0.01), np.arange(bb_min, bb_max, 0.01), 'g')

            slope, intercept, _, _, _ = linregress(self._scatter_info.get_x_data(ind, dimension=self.dimension),
                                                   self._scatter_info.get_y_data(ind, dimension=self.dimension))
            line_x_range = np.arange(bb_min, bb_max, 0.01)
            line_y_range = line_x_range * slope + intercept

            pos = np.where(np.logical_and(bb_min <= line_y_range, line_y_range<= bb_max))
            line_x_range = line_x_range[pos]
            line_y_range = line_y_range[pos]

            plt.plot(line_x_range, line_y_range, 'r')

            plt.xlabel(self._scatter_info.get_x_label(ind, dimension=self.dimension))
            plt.ylabel(self._scatter_info.get_y_label(ind, dimension=self.dimension))

            if self.show_titles:
                plt.title(self._scatter_info.get_plot_title(ind, dimension=self.dimension))

            plt.subplots_adjust(hspace=.3)

        self._figure.suptitle(self._scatter_info.get_plot_title(dimension=self.dimension))

        self._figure.canvas.draw()

        mng = plt.get_current_fig_manager()
        mng.canvas.set_window_title(self._scatter_info.get_plot_title(dimension=self.dimension))

    def _get_bounding_box(self):
        min_list = []
        max_list = []

        for ind in range(self._scatter_info.get_nmr_plots()):
            line_min = min(np.min(self._scatter_info.get_x_data(ind, dimension=self.dimension)),
                           np.min(self._scatter_info.get_y_data(ind, dimension=self.dimension)))
            line_max = max(np.max(self._scatter_info.get_x_data(ind, dimension=self.dimension)),
                           np.max(self._scatter_info.get_y_data(ind, dimension=self.dimension)))

            min_list.append(line_min)
            max_list.append(line_max)

        minimum = float(np.around(min(min_list), decimals=2))
        maximum = float(np.around(max(max_list), decimals=2))

        return max(minimum - 0.1, 0), min(maximum + 0.1, 1)


class _DiscreteSlider(Slider):
    """A matplotlib slider widget with discrete steps."""

    def __init__(self, *args, **kwargs):
        """Identical to Slider.__init__, except for the "increment" kwarg.
        "increment" specifies the step size that the slider will be discritized
        to."""
        self.inc = kwargs.pop('increment', 0.25)
        Slider.__init__(self, *args, **kwargs)

    def set_max(self, new_max):
        orig_val = self.val
        self.set_val(self.valmin)

        self.valmax = new_max
        self.ax.set_xlim((self.valmin, self.valmax))

        if orig_val >= new_max:
            self.set_val((new_max + self.valmin) / 2.0)
        else:
            self.set_val(orig_val)

    def set_val(self, val):
        discrete_val = int(val / self.inc) * self.inc
        # We can't just call Slider.set_val(self, discrete_val), because this
        # will prevent the slider from updating properly (it will get stuck at
        # the first step and not "slide"). Instead, we'll keep track of the
        # the continuous value as self.val and pass in the discrete value to
        # everything else.
        xy = self.poly.xy
        xy[2] = discrete_val, 1
        xy[3] = discrete_val, 0
        self.poly.xy = xy
        self.valtext.set_text(self.valfmt % discrete_val)
        if self.drawon:
            self.ax.figure.canvas.draw()
        self.val = val
        if not self.eventson:
            return
        for cid, func in self.observers.iteritems():
            func(discrete_val)
