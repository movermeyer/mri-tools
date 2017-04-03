import glob
import os
import re


__author__ = 'Robbert Harms'
__date__ = "2015-05-04"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"

"""
This example script shows how to automate fetching the read out time (in seconds) from the xml file.

This will calculate the following:
read_out_time = echo_spacing * ((base-resolution [* phase oversampling, if used] *
                                (partial Fourier Factor /  GRAPPAFactor))-1)

As an example, let's assume that your matrix has 128 rows along the phase encoding direction,
you used a partial Fourier factor of 6/8 and a GRAPPA factor of 2, then the multiplication factor is 47.
Suppose then that the echo spacing is 0.0005 s which makes the total readout time 0.0235s.

It is assumed that the Phase oversampling is in percentages and is calculated as::

    (100 + phase_oversampling_percentage)/100

If you do not have the .xml file with the scanner settings you can also manually create the read out time files::

    <volume_name>.read_out_times.txt

So, suppose you have the scan "006_mb_diff_6k_113dirs_2mm_AP_MB2G2.nii", then you should create a file named::

    006_mb_diff_6k_113dirs_2mm_AP_MB2G2.read_out_times.txt

Containing on one line a floating point number for the read out times in seconds. Do this for every measurement.
"""

def calculate_read_out_time():
    from mri_tools.scanner_settings.parsers.siemens import PrismaInfoReader

    settings_file = '/home/robbert/programming/python/mdt/bin/multite_20150417_alard/20150417_RobbertMultiTE.xml'
    uncorrected_dir = '/home/robbert/programming/python/mdt/bin/multite_20150417_alard/1_uncorrected/'

    info_reader = PrismaInfoReader(settings_file)
    read_out_times = info_reader.get_read_out_time()

    for session_name, value in read_out_times.items():
        session_name = re.sub(r'[^a-zA-Z0-9]', '', session_name)

        with open(os.path.join(uncorrected_dir, session_name + '.read_out_times.txt'), 'w') as f:
            f.write(str(value))


def write_read_out_time(read_out_time):
    uncorrected_dir = '/home/robbert/phd-data/mdt_example_data/niis/'

    for f in glob.glob(os.path.join(uncorrected_dir, '*.nii*')):
        basename = os.path.basename(f)
        basename = basename.split('.')[0]

        with open(os.path.join(uncorrected_dir, basename + '.read_out_times.txt'), 'w') as write_file:
            write_file.write(str(read_out_time))

write_read_out_time(0.0731289)
