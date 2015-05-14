from collections import OrderedDict
import os
import re
import xml.etree.ElementTree as ET
from mri_tools.scanner_settings.parsers.common import ScannerSettingsParser, InfoReader

__author__ = 'Robbert Harms'
__date__ = "2015-05-04"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class PrismaInfoReader(InfoReader):

    def __init__(self, settings_file):
        """Create a new info reader that can read specifics from the given input file.

        This will first construct a Parser, then using that parser reads the values to calculate the read out time.

        Args:
            parser (ScannerSettingsParser): The parser to use
            settings_file (str): The file to use for the parsing
        """
        self._settings_file = settings_file

        extension = os.path.splitext(self._settings_file)[1].lower()[1:]

        if extension == 'pdf':
            raise ValueError('PDF is not supported, please use XML or TXT')
        elif extension == 'xml':
            self._parser = PrismaXMLParser(self._settings_file)
        elif extension == 'txt':
            self._parser = PrismaTXTParser(self._settings_file)
        else:
            raise ValueError('Could not identify the file type of the given settings file.')

    def get_read_out_time(self):
        """Get the read out time from the settings file, in seconds.

        This will calculate the following:
        echo_spacing * ((base-resolution [* phase oversampling, if used] * (partial Fourier Factor /  GRAPPAFactor))-1)

        As an example, let's assume that your matrix has 128 rows along the phase encoding direction,
        you used a partial Fourier factor of 6/8 and a GRAPPA factor of 2,
        the right number is 47.

        If the echo spacing is 0.0005 s, then your total readout time is 0.0235s.

        It is assumed that the Phase oversampling is in percentages and is calculated as:
            (100 + phase_oversampling_percentage)/100

        Returns:
            float: The read out time in seconds.
        """
        echo_spacings = self._get_echo_spacing()
        base_resolutions = self._get_base_resolution()
        phase_oversamplings = self._get_phase_oversampling()
        phase_partial_fouriers = self._get_phase_partial_fourier()
        grappa_factors = self._get_accel_factor_pe()

        dicts = [echo_spacings, base_resolutions, phase_oversamplings, phase_partial_fouriers, grappa_factors]

        read_out_times = {}
        for key in echo_spacings.keys():
            if all([key in d for d in dicts]):
                phase_oversampling = 1 + (phase_oversamplings[key] * 1.0e-2)
                read_out_time = echo_spacings[key] * (base_resolutions[key] *
                                                      phase_oversampling *
                                                      (phase_partial_fouriers[key] / grappa_factors[key]) - 1)
                read_out_times.update({key: read_out_time})

        return read_out_times

    def _get_echo_spacing(self):
        """Get the echo spacing from the settings file. This will convert the input from ms to seconds.

        This assumes that the input value is in ms (which it is in the siemens prisma). It will then scale by 1e-3 to
        get to seconds.

        This will read the key 'Echo spacing'.

        Returns:
            dict: A dictionary with as keys the session names and as values the echo spacings
        """
        values = self._parser.get_value('Echo spacing')
        return {k: float(re.sub(r'[^0-9.]', '', v))*1e-3 for k, v in values.items() if v is not None}

    def _get_epi_factor(self):
        """Get the epi factor spacing from the settings file

        This will read the key 'EPI factor'.

        Returns:
            dict: A dictionary with as keys the session names and as values the epi factors
        """
        values = self._parser.get_value('EPI factor')
        return {k: float(re.sub(r'[^0-9.]', '', v)) for k, v in values.items() if v is not None}

    def _get_accel_factor_pe(self):
        """Get the epi factor spacing from the settings file

        This will read the key 'Accel. factor PE'.

        Returns:
            dict: A dictionary with as keys the session names and as values the acceleration factors
        """
        values = self._parser.get_value('Accel. factor PE')
        return {k: float(re.sub(r'[^0-9.]', '', v)) for k, v in values.items() if v is not None}

    def _get_base_resolution(self):
        """Get the base resolution from the settings file

        This will read the key 'Base resolution'.

        Returns:
            dict: A dictionary with as keys the session names and as values the acceleration factors
        """
        values = self._parser.get_value('Base resolution')
        return {k: int(re.sub(r'[^0-9.]', '', v)) for k, v in values.items() if v is not None}

    def _get_phase_oversampling(self):
        """Get the base resolution from the settings file

        This will read the first key 'Phase oversampling' found (in general the one of paragraph 'Routine').

        Returns:
            dict: A dictionary with as keys the session names and as values the acceleration factors
        """
        values = self._parser.get_value('Phase oversampling')
        return {k: float(re.sub(r'[^0-9.]', '', v)) for k, v in values.items() if v is not None}

    def _get_phase_partial_fourier(self):
        """Get the phase partial fourier from the settings file

        This will read the first key 'Phase partial Fourier' found.

        Returns:
            dict: A dictionary with as keys the session names and as values the acceleration factors
        """
        values = self._parser.get_value('Phase partial Fourier')
        return {k: self._parse_division(re.sub(r'[^0-9./]', '', v)) for k, v in values.items() if v is not None}

    def _parse_division(self, simple_division):
        """Parses a string with a simple division like '6/8' to a float.

        Args:
            simple_division (str): A string with a simple division like '6/8'.

        Returns:
            float: The string converted to float
        """
        if simple_division == '':
            return None
        parts = simple_division.split('/')
        return float(parts[0]) / float(parts[1])


class PrismaXMLParser(ScannerSettingsParser):

    def __init__(self, settings_file):
        """Create an XML parser for the given settings file.

        The settings file is supposed to come from a Siemens Prisma 3T scanner.

        Args:
            settings_file (str): The filepath to the settings file.
        """
        self._settings_file = settings_file

    def get_value(self, key):
        tree = ET.parse(self._settings_file)
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


class PrismaTXTParser(ScannerSettingsParser):

    def __init__(self, settings_file):
        """Create an TXT parser for the given settings file.

        The settings file is supposed to come from a Siemens Prisma 3T scanner.

        Args:
            settings_file (str): The filepath to the settings file.
        """
        self._settings_file = settings_file

    def get_value(self, key):
        sessions = self._parse_file()

        values = {}
        for session_id, session_properties in sessions.items():
            for paragraph in session_properties.values():
                for prop, value in paragraph.items():
                    if prop == key:
                        values.update({session_id: value})

        return values

    def _parse_file(self):
        """Parse the file into a layer of dictionaries.

        Returns:
            dict: with dicts containing dicts.
                The first level is for the sessions, the second for the paragraphs the third layer for the properties.
        """
        sessions = self._parse_sessions()
        return {k: self._parse_values(v) for k, v in sessions.items()}

    def _parse_sessions(self):
        """First level of parsing, dividing the txt file into sessions.

        Returns:
            dict: with as keys the session names and as values the raw string in that session.
        """
        in_session = False
        in_session_header = False

        sessions = {}
        lines = []

        with open(self._settings_file) as f:
            for line in f:
                line = line.rstrip()
                if line[:10] == '-' * 10:
                    if in_session:
                        in_session = False
                        in_session_header = True
                        del lines[len(lines)-1]
                        lines = []

                    elif in_session_header:
                        in_session = True
                        in_session_header = False

                        if lines[0].strip() == 'Table Of Contents':
                            return sessions
                        else:
                            header_parts = lines[0].split('\\')
                            header = header_parts[len(header_parts) - 1]
                            sessions.update({header: []})
                            lines = sessions[header]
                    else:
                        in_session_header = True
                else:
                    if in_session:
                        lines.append(line)
                    elif in_session_header:
                        lines.append(line)
        return sessions

    def _parse_values(self, lines):
        """Parse the values from one session.

        Returns:
            dict: with as keys the paragraph names and as values a dictionary. This second dictionary is for the
                properties and has a key the property name and as value the property value.

                Example dict:
                    {'Properties': {'Prio Recon': 'Off', ...}, ...}
        """
        paragraphs = OrderedDict()
        properties = OrderedDict()

        for line in lines:
            if line[:5] != ' ' * 5:
                paragraphs.update({line: OrderedDict()})
                properties = paragraphs[line]
            else:
                line = line.strip()
                if line != '':
                    key = ''
                    space_count = 0

                    for char in line:
                        if char == ' ':
                            space_count += 1
                            if space_count > 1:
                                break
                        else:
                            space_count = 0
                        key += char

                    value = line[len(key):].strip()
                    key = key.strip()
                    properties.update({key: value})

        return paragraphs