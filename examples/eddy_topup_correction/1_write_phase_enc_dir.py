import glob
import os

__author__ = 'Robbert Harms'
__date__ = "2015-05-04"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


uncorrected_dir = '/home/robbert/programming/python/mdt/bin/multite_20150417_alard/1_uncorrected/'
phase_enc_dirs = ['AP', 'PA', 'LR', 'RL', 'DS', 'SD', 'SI', 'IS', 'HF', 'FH']

for f in glob.glob(os.path.join(uncorrected_dir, '*.nii.gz')):
    basename = os.path.basename(f)
    basename = basename.split('.')[0]

    phase_enc_dir = 'PA'
    for ped in phase_enc_dirs:
        if ped in basename:
            phase_enc_dir = ped

    with open(os.path.join(uncorrected_dir, basename + '.phase_enc_dir.txt'), 'w') as write_file:
        write_file.write(phase_enc_dir)