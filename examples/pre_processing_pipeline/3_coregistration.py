from mri_tools.registration.coregister import coregister

__author__ = 'Robbert Harms'
__date__ = "2015-05-14"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


input_dir = '/home/robbert/programming/python/mdt/bin/multite_20150417_alard/3_corrected/'
tmp_dir = '/home/robbert/programming/python/mdt/bin/multite_20150417_alard/4_registered_tmp/'
output_dir = '/home/robbert/programming/python/mdt/bin/multite_20150417_alard/5_registered/'

items = ['35dir', '53dir', '53dir_TEmax', '35dir_TEmax']

coregister(items, input_dir, tmp_dir, output_dir)
