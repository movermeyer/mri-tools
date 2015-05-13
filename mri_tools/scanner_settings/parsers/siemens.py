import re
import xml.etree.ElementTree as ET
from mri_tools.scanner_settings.parsers.common import ScannerSettingsParser

__author__ = 'Robbert Harms'
__date__ = "2015-05-04"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class InfoReader(object):

    def __init__(self, parser, settings_file):
        """Create a new info reader that can read specifics from the given input file.

        Args:
            parser (ScannerSettingsParser): The parser to use
            settings_file (str): The file to use for the parsing
        """
        self._parser = parser
        self._settings_file = settings_file

    def get_echo_spacing(self):
        """Get the echo spacing from the settings file. This will convert the input from ms to seconds.

        This assumes that the input value is in ms (which it is in the siemens prisma). It will then scale by 1e-3 to
        get to seconds.

        This will read the key 'Echo spacing'.

        Returns:
            dict: A dictionary with as keys the session names and as values the echo spacings
        """
        values = self._parser.get_value(self._settings_file, 'Echo spacing')
        return {k: float(re.sub(r'[^0-9.]', '', v))*1e-3 for k, v in values.items() if v is not None}

    def get_epi_factor(self):
        """Get the epi factor spacing from the settings file

        This will read the key 'EPI factor'.

        Returns:
            dict: A dictionary with as keys the session names and as values the epi factors
        """
        values = self._parser.get_value(self._settings_file, 'EPI factor')
        return {k: float(re.sub(r'[^0-9.]', '', v)) for k, v in values.items() if v is not None}

    def get_accel_factor_pe(self):
        """Get the epi factor spacing from the settings file

        This will read the key 'Accel. factor PE'.

        Returns:
            dict: A dictionary with as keys the session names and as values the acceleration factors
        """
        values = self._parser.get_value(self._settings_file, 'Accel. factor PE')
        return {k: float(re.sub(r'[^0-9.]', '', v)) for k, v in values.items() if v is not None}


class PrismaXMLParser(ScannerSettingsParser):

    def get_value(self, settings_file, key):
        """This will get a value for the given key for every session in the settings file.

        Args:
            settings_file (str): the filename to the settings file
            key (str): the key for which to get the value for every session.

        Returns:
            dict: with as keys the session name and as value the value for the key for that session
        """
        tree = ET.parse(settings_file)
        root = tree.getroot()

        values = {}
        for child in root:
            if child.tag == 'PrintProtocol':
                header_property = child[0][1][0].text
                file_path = header_property.split('\\')
                session_name = file_path[len(file_path)-1]

                value = self._find_prot_parameter(child[0], key)
                values.update({session_name: value})

        return values

    def _find_prot_parameter(self, root, label_name):
        for child in root:
            if child.tag == 'ProtParameter':
                if child[0].text == label_name:
                    return child[1].text
            sub_search = self._find_prot_parameter(child, label_name)
            if sub_search is not None:
                return sub_search
        return None

