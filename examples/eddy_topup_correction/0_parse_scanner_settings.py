import os
import re
from mri_tools.scanner_settings.parsers.siemens import PrismaXMLParser, InfoReader

__author__ = 'Robbert Harms'
__date__ = "2015-05-04"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


settings_file = '/home/robbert/programming/python/mdt/bin/multite_20150417_alard/20150417_RobbertMultiTE.xml'
uncorrected_dir = '/home/robbert/programming/python/mdt/bin/multite_20150417_alard/1_uncorrected/'

info_reader = InfoReader(PrismaXMLParser(), settings_file)
epi_factors = info_reader.get_epi_factor()
echo_spacings = info_reader.get_echo_spacing()
accel_factors = info_reader.get_accel_factor_pe()


def write_values(value_dict, output_dir, extension):
    for session_name, value in value_dict.items():
        session_name = re.sub(r'[^a-zA-Z0-9]', '', session_name)

        with open(os.path.join(output_dir, session_name + extension), 'w') as f:
            f.write(str(value))

write_values(epi_factors, uncorrected_dir, '.epi_factor.txt')
write_values(echo_spacings, uncorrected_dir, '.echo_spacing.txt')
write_values(accel_factors, uncorrected_dir, '.acc_fac_pe.txt')