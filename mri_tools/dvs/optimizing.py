from mri_tools.dvs.base import DVSDirectionTable
import numpy as np

__author__ = 'Robbert Harms'
__date__ = "2015-04-23"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_hot_spots(dvs_table, max_diff, max_gradient):
    """Get the indices of the hotspots.

    Hotspots are positions where there are two gradients with large components next to each other.

    To find them, we first take from each gradient the maximum element 'maxv'. We then take the difference
    between each value and its (right) neighbour 'diff'.

    Hotspots are then defined as positions where 'diff' is lower than the maximum difference and the 'maxv' is
    larger than the maximum allowable gradient.

    Args:
        dvs_table (ndarray): a 3d array with the gradients.
        max_diff (double): The maximum allowable gradient scale.
        max_gradient (double): The maximum allowable gradient

    Returns:
        list: A list with indices for hotspot positions. The list is ordered by gradient amplitude. High to low.
    """
    diff = np.abs(np.diff(np.max(dvs_table, axis=1)))
    diff = np.append(diff, 0)
    maxv = np.max(dvs_table, axis=1)

    hot_spots = (diff < max_diff) * (maxv > max_gradient)
    indices = reversed(np.argsort(maxv).tolist())

    return_list = []
    for ind in indices:
        if hot_spots[ind]:
            return_list.append(ind)
    return return_list


def get_cold_spots(dvs_table, max_gradient):
    """Get the indices of the coldspots.

    Coldspots are positions where there are two gradients with small components next to each other.

    To find them, we first take from each gradient the maximum element 'maxv'. We then take the difference
    between each value and its (right) neighbour 'diff'.

    Coldspots are then defined as positions where 'diff' is higher than the minimum difference and the 'maxv' is
    smaller than the maximum allowable gradient.

    Args:
        dvs_table (ndarray): a 3d array with the gradients.
        max_diff (double): The maximum allowable gradient scale.
        max_gradient (double): The maximum allowable gradient

    Returns:
        list: A list with indices for low spot positions. The list is in order of occurrence.
    """
    maxv = np.max(dvs_table, axis=1)
    pos = maxv < max_gradient

    return_list = []
    for i, e in enumerate(pos.tolist()):
        if e:
            return_list.append(i)
    return return_list


def swap(dvs_table, a, b):
    """Procedure to swap two gradients in place.

    Args:
        dvs_table (ndarray): a 3d array with the gradients.
        a (int): The first position
        a (int): The second position

    Returns:
        ndarray: The swapped table to allow functional calls.
    """
    tmp = np.copy(dvs_table[a])
    dvs_table[a] = dvs_table[b]
    dvs_table[b] = tmp
    return dvs_table


def remove_hotspots(dvs_table, max_diff, max_gradient):
    """Procedure to recursively remove the hotpots. This works in place.

    Hotspots are positions where there are two gradients with large components next to each other. This function tries
    to remove them.

    Note that is may not be possible to remove them given the constraints. Completely optimizing the table to reduce all
    distances is an NP-Complete problem.

    Args:
        dvs_table (ndarray): a 3d array with the gradients.
        max_diff (double): The maximum allowable gradient scale.
        max_gradient (double): The maximum allowable gradient
    """
    return _recursive_remove_hotspots(dvs_table, max_diff, max_gradient, depth_limit=10)


def _recursive_remove_hotspots(dvs_table, max_diff, max_gradient, depth_limit=10, current_depth=0):
    hot_spots = get_hot_spots(dvs_table, max_diff=max_diff, max_gradient=max_gradient)

    if current_depth == depth_limit:
        return dvs_table

    for hot_spot in hot_spots:
        cold_spot_max_gradient = 0
        cold_spots = get_cold_spots(dvs_table, max_gradient=cold_spot_max_gradient)
        while not cold_spots:
            max_gradient += 0.1
            cold_spots = get_cold_spots(dvs_table, max_gradient=cold_spot_max_gradient)
        dvs_table = swap(dvs_table, hot_spot, cold_spots[0])

    return _recursive_remove_hotspots(dvs_table, max_diff, max_gradient, depth_limit=depth_limit,
                                      current_depth=current_depth + 1)