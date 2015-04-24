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
        max_gradient_dir = np.max(table, axis=1)
        low_high_ind = np.argsort(max_gradient_dir).tolist()
        high_low_ind = list(reversed(low_high_ind))

        interwoven = low_high_ind + high_low_ind
        interwoven[::2] = low_high_ind
        interwoven[1::2] = high_low_ind

        new_table = np.zeros_like(table)
        for i, ind in enumerate(interwoven):
            if i == table.shape[0]:
                break
            new_table[i, :] = table[ind, :]

        return new_table