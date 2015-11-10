import copy
import numpy as np
import csv
import os
import itertools
from mri_tools.plots.layouts import LowerTriangleGridLayout
from mri_tools.plots.scatter_plots import SimpleScatterData, ScatterDataInfo, ScatterPlots, MultiROIScatterPlots

__author__ = 'Robbert Harms'
__date__ = "2015-11-06"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ScatterROIs(object):

    def __init__(self, base_dir, column_info_file=None):
        """Functions for displaying the CSV output in various ways.

        Args;
            base_dir (str): the directory containing the CSV files
        """
        self._base_dir = base_dir
        self._skip_initial_rows = 1
        self._column_info_file = column_info_file or os.path.join(self._base_dir, 'column_info.txt')
        self._roi_titles = self._load_roi_titles()

    def plot_single_roi(self, map_names, roi_index, map_titles=None, to_file=None, block=True):
        """Show the scatter plots of all the combinations of the given maps.

        Args:
            map_names (list of str): the list of map names we want to use for the scatterplots. It is supposed
                that the CSV files are in the base dir and have the extension .csv
            roi_index (int): the column we want to show (effectively the roi we are interested in)

        """
        scatter_info = self._load_scatter_data(map_names, map_titles)
        plots = ScatterPlots(scatter_info, placement=LowerTriangleGridLayout(len(map_names) - 1))
        plots.show(dimension=roi_index, show_titles=False, to_file=to_file, block=block)

    def plot_multi_roi(self, map_names, roi_indices, map_titles=None, dimension_titles=None, to_file=None, block=True,
                       legend_plot_options=None):
        """Show the scatter plots of all the combinations of the given maps.

        Args:
            map_names (list of str): the list of map names we want to use for the scatterplots. It is supposed
                that the CSV files are in the base dir and have the extension .csv
            roi_indices (int): the columns (roi indices) we want to show
            dimension_titles (dict): if given, per roi index the specific title to overwrite the default title
        """
        scatter_info = self._load_scatter_data(map_names, map_titles, dimension_titles)
        plots = MultiROIScatterPlots(scatter_info, placement=LowerTriangleGridLayout(len(map_names) - 1))
        plots.show(roi_indices, show_titles=False, to_file=to_file, block=block,
                   legend_plot_options=legend_plot_options)

    def _load_roi_titles(self):
        roi_titles = []
        with open(self._column_info_file) as f:
            start = itertools.dropwhile(lambda l: l.lower().lstrip().startswith('#'), f)
            column_reader = list(csv.reader(start, delimiter=',', quotechar='"'))
            for row_ind, row in enumerate(column_reader):
                if row_ind >= self._skip_initial_rows:
                    roi_titles.append(row[2] + ' (' + row[3] + ')')
        return roi_titles

    def _load_scatter_data(self, map_names, map_titles=None, dimension_titles=None):
        map_titles = map_titles or {}
        scatter_data_list = []
        for map_names in itertools.combinations(map_names, r=2):
            data = []
            labels = []
            for map_name in map_names:
                path = os.path.join(self._base_dir, map_name + '.csv')
                csv_data = np.genfromtxt(path, delimiter=',')[:, self._skip_initial_rows:]
                data.append(csv_data)

                if map_name in map_titles:
                    labels.append(map_titles[map_name])
                else:
                    labels.append(map_name)

            arg_list = []
            arg_list.extend(data)
            arg_list.extend(labels)
            arg_list.append(' - '.join(labels))

            scatter_data_list.append(SimpleScatterData(*arg_list))

        titles = self._roi_titles
        if dimension_titles:
            titles = list(copy.copy(self._roi_titles))
            for ind, title in dimension_titles.items():
                titles[ind] = title

        return ScatterDataInfo(scatter_data_list, titles)