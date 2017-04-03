import multiprocessing
from mri_tools.topup_eddy.common import run_pre_processing

__author__ = 'Robbert Harms'
__date__ = "2015-05-04"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"

# input_dir = '/home/robbert/programming/python/mdt/bin/multite_20150417_alard/1_uncorrected/'
# tmp_dir = '/home/robbert/programming/python/mdt/bin/multite_20150417_alard/2_corrected_tmp/'
# output_dir = '/home/robbert/programming/python/mdt/bin/multite_20150417_alard/3_corrected/'
#
# items = (('mbdiffb1k35dirs2mmAPMB2G2', 'mbdiffb1k35dirs2mmPAMB2G2', '35dir'),
#          ('mbdiffb2k53dirs2mmAPMB2G2', 'mbdiffb2k53dirs2mmPAMB2G2', '53dir'),
#          ('mbdiffb2k53dirs2mmAPMB2G2TEmax', 'mbdiffb2k53dirs2mmPAMB2G2TEmax', '53dir_TEmax'),
#          ('mbdiffb1k35dirs2mmAPMB2G2TEmax', 'mbdiffb1k35dirs2mmPAMB2G2TEmax', '35dir_TEmax'))


input_dir = '/home/robbert/phd-data/mdt_example_data/niis/'
tmp_dir = '/home/robbert/phd-data/mdt_example_data/2_corrected_tmp/'
output_dir = '/home/robbert/phd-data/mdt_example_data/3_corrected/'

items = (('mbdiff1k35dirs1p5mmAPMB2G2s010a001', 'mbdiffb206dirs1p5mmPAMB2G2s012a001', '35dir_1p5mm'),
         ('mbdiff2k66dirs1p5mmAPMB2G2s009a001', 'mbdiffb206dirs1p5mmPAMB2G2s012a001', '66dir_1p5mm'),
         ('mbdiff6k113dirs2mmAPMB2G2s006a001', 'mbdiffb206dirs2mmPAMB2G2s008a001', '113dir_2mm'),)

def process(item):
    run_pre_processing(item[0], item[1], item[2], input_dir, tmp_dir, output_dir)

pool = multiprocessing.Pool()
pool.map(process, items)
