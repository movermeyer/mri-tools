from mri_tools.common import create_mask_from_mean_unweighted

__author__ = 'Robbert Harms'
__date__ = "2015-05-14"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


item_dir = '/home/robbert/programming/python/mdt/bin/multite_20150417_alard/5_registered/'
item = 'combined'

create_mask_from_mean_unweighted(item_dir, item)


