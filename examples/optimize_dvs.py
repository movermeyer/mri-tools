from mri_tools.dvs.io import read_dvs, write_dvs
from mri_tools.dvs.restructuring import LowHighOptimizer
from mri_tools.dvs.plot import DVSPlot

__author__ = 'Robbert Harms'
__date__ = "2015-04-23"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"

dvs_0 = read_dvs('/home/robbert/Documents/phd/gradient_files/MBIC_DiffusionVectors.dvs')
dvs_1 = read_dvs('/home/robbert/Documents/phd/gradient_files/MBIC_DiffusionVectors_v1.dvs')
dvs_2 = read_dvs('/home/robbert/Documents/phd/gradient_files/MBIC_DiffusionVectors_v2.dvs')

print(dvs_0.get_overview_representation())

table_ind = 18
dvs_plot = DVSPlot()
dvs_plot.draw_table_at_index(dvs_0, table_ind)
dvs_plot.draw_table_at_index(dvs_1, table_ind)
dvs_plot.draw_table_at_index(dvs_2, table_ind)


dvs_plot.plot_block()


#
# optimizer = LowHighOptimizer()
# optimized_dvs = optimizer.optimize(dvs)
#
# dvs_plot.draw_table_at_index(optimized_dvs, table_ind)
#
# optimized_table = optimizer.optimize(dvs.dvs_tables[table_ind].table)
# dvs_plot.draw_table(optimized_table)
#
# dvs_plot.plot_block()
#
# write_dvs('/home/robbert/Documents/phd/gradient_files/MBIC_DiffusionVectors_v1.dvs', optimized_dvs)