import copy
from mri_tools.dvs.base import DVSDirectionTable, DVS
import numpy as np

__author__ = 'Robbert Harms'
__date__ = "2015-04-23"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AbstractDVSLayoutOptimizer(object):

    def optimize(self, dvs):
        """Optimize the layout of the given DVS object.

        The problem the optimizer tries to solve is the following. The Siemens Prisma 3T scanners crashes
        when there are two gradients after each other which have a strong amplitude on one of the axis, both
        of the same polarity. For example: (0, 0, 1) and (0, 1, 0) will crash the scanner.

        To solve this issue implement this class and implement a heuristic.

        Args:
            dvs (DVS, DVSDirectionTable or ndarray): Either a complete DVS object, a DVSDirectionTable object or
                a ndarray with shape (n, 3). If a DVS object is given all tables in the DVS are optimized.

        Returns:
            A copy of the original object but then optimized.
        """
        new_dvs = copy.deepcopy(dvs)
        if isinstance(dvs, DVS):
            for dvs_table in new_dvs.dvs_tables:
                dvs_table.table = self._optimize_table(dvs_table.table)
        elif isinstance(dvs, DVSDirectionTable):
            new_dvs.table = self._optimize_table(new_dvs)
        else:
            return self._optimize_table(new_dvs)

        return new_dvs

    def _optimize_table(self, table):
        """Optimize a gradient table.

        This optimization can take place inplace. This function is supposed to be subclassed.

        Args:
            table (ndarray): The actual table to be optimized.

        Returns:
            ndarray: The optimized array.
        """
        return table


class LowHighOptimizer(AbstractDVSLayoutOptimizer):

    def _optimize_table(self, table):
        max_gradient_dirs = np.max(table, axis=1)
        interwoven = self._get_ordered_indices(max_gradient_dirs)

        print(self._get_ordered_indices(np.array([1, 2, 3, 4, 5, 6, 7])))

        new_table = np.zeros_like(table)
        for i, ind in enumerate(interwoven):
            if i == table.shape[0]:
                break
            new_table[i, :] = table[ind, :]

        return new_table

    def _get_ordered_indices(self, max_gradient_dirs):
        low_high_ind = np.argsort(max_gradient_dirs)
        return self._split_interweave(low_high_ind, number_of_interweaving=1)

    def _split_interweave(self, input_vector, number_of_interweaving=0, current_depth=0):
        """Recursively splits the input vector into two halves and interweaves them.

        A simple example is the following:
            input: [0, 1, 2, 3] number_of_interweaving=0
            output: [0, 2, 1, 3]

        A larger example:
            input [0, 1, 2, 3, 4, 5, 6]
            output: [0, 6, 1, 5, 2, 4, 3] if you set number of interweaving to 0
            or: [0, 5, 3, 6, 1, 4, 2] if you set the number of interweaving to 1.

        Args:
            input_vector (list or ndarray): The list with items we want to interweave. Assumes it is sorted.
            number_of_interweaving (int): How many recursive times we want to interleave
            current_depth (int): Parameter to limit the recursion.

        Returns:
            A list with the indices interwoven.
        """
        halves = np.array_split(np.array(input_vector), 2)
        halves = [v.tolist() for v in halves]

        if len(halves) < 2:
            return input_vector

        if current_depth < number_of_interweaving:
            halves = [self._split_interweave(v, number_of_interweaving, current_depth + 1) for v in halves]

        first_half = halves[0]
        second_half = list(reversed(halves[1]))

        interwoven = first_half + second_half
        interwoven[::2] = first_half
        interwoven[1::2] = second_half

        return interwoven