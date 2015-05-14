import os
import re
from mri_tools.scanner_settings.parsers.siemens import PrismaInfoReader

__author__ = 'Robbert Harms'
__date__ = "2015-05-04"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


settings_file = '/home/robbert/programming/python/mdt/bin/multite_20150417_alard/20150417_RobbertMultiTE.xml'
uncorrected_dir = '/home/robbert/programming/python/mdt/bin/multite_20150417_alard/1_uncorrected/'

info_reader = PrismaInfoReader(settings_file)
read_out_times = info_reader.get_read_out_time()


def write_values(value_dict, output_dir, extension):
    for session_name, value in value_dict.items():
        session_name = re.sub(r'[^a-zA-Z0-9]', '', session_name)

        with open(os.path.join(output_dir, session_name + extension), 'w') as f:
            f.write(str(value))

write_values(read_out_times, uncorrected_dir, '.read_out_times.txt')
