from mri_tools.dwi_pre_processing.common import apply_corrections

__author__ = 'Robbert Harms'
__date__ = "2015-05-04"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"

input_dir = '/home/robbert/programming/python/mdt/bin/multite_20150417_alard/1_uncorrected/'
tmp_dir = '/home/robbert/programming/python/mdt/bin/multite_20150417_alard/2_corrected_tmp/'
output_dir = '/home/robbert/programming/python/mdt/bin/multite_20150417_alard/3_scans/'

epis = ['mbdiffb1k35dirs2mmAPMB2G2', 'mbdiffb2k53dirs2mmAPMB2G2',
        'mbdiffb2k53dirs2mmAPMB2G2TEmax', 'mbdiffb1k35dirs2mmAPMB2G2TEmax']

alt_epis = ['mbdiffb1k35dirs2mmPAMB2G2', 'mbdiffb2k53dirs2mmPAMB2G2',
            'mbdiffb2k53dirs2mmPAMB2G2TEmax', 'mbdiffb1k35dirs2mmPAMB2G2TEmax']

apply_corrections(epis, alt_epis, input_dir, tmp_dir, output_dir)
