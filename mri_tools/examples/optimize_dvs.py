from mri_tools.dvs.io import read_dvs, write_dvs
from mri_tools.dvs.optimizing import LowHighOptimizer
from mri_tools.dvs.plot import DVSPlot
import numpy as np

__author__ = 'Robbert Harms'
__date__ = "2015-04-23"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"

dvs = read_dvs('/home/robbert/Documents/phd/gradient_files/MBIC_DiffusionVectors.dvs')

# print(dvs.get_overview_representation())

table_ind = 0

dvs_plot = DVSPlot()
dvs_plot.draw_table_at_index(dvs, table_ind)

optimizer = LowHighOptimizer()
optimized_dvs = optimizer.optimize(dvs)

# dvs_plot.draw_table_at_index(optimized_dvs, table_ind)

optimized_table = optimizer.optimize(dvs.dvs_tables[table_ind].table)
dvs_plot.draw_table(optimized_table)

dvs_plot.plot_block()

write_dvs('/home/robbert/Documents/phd/gradient_files/MBIC_DiffusionVectors_v2.dvs', optimized_dvs)