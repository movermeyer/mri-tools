from mri_tools.dvs.io import read_dvs, write_dvs
from mri_tools.dvs.optimizing import get_hot_spots, get_cold_spots, swap, remove_hotspots
from mri_tools.dvs.plot import DVSPlot
import numpy as np

__author__ = 'Robbert Harms'
__date__ = "2015-04-23"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"

dvs = read_dvs('/home/robbert/Documents/phd/gradient_files/MBIC_DiffusionVectors.dvs')

# print(dvs.get_overview_representation())

table_nmr = 17
table = dvs.tables[table_nmr].table

dvs_plot = DVSPlot()
dvs_plot.draw_table(table)

table = remove_hotspots(table, max_diff=0.1, max_gradient=0.95)
hot_spots = get_hot_spots(table, max_diff=0.1, max_gradient=0.95)

dvs_plot.draw_table(table)
dvs_plot.plot_block()

write_dvs('/home/robbert/Documents/phd/gradient_files/MBIC_DiffusionVectors_v2.dvs', dvs)