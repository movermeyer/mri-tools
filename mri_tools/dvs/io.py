import re
import numpy as np
from mri_tools.dvs.base import DVS, DVSDirectionTable

__author__ = 'Robbert Harms'
__date__ = "2015-04-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class DVSParser(object):

    def __init__(self):
        pass

    def parse(self, dvs_str):
        """Parse a string and create a new DVS file object.

        Args:
            dvs_str (str): The string containing a DVS file.

        Returns:
            DVS: A DVS object representation from the given string.
        """
        dvs_lines = dvs_str.split('\n')
        dvs_lines = self._clean_lines(dvs_lines)

        comments, dvs_lines = self._consume_top_comments(dvs_lines)
        tables = []

        while dvs_lines:
            table, dvs_lines = self._consume_table(dvs_lines)
            tables.append(table)

        return DVS(comments, tables)

    def _parse_table(self, dvs_lines):
        comments = ''
        count = 0
        for line in dvs_lines:
            if len(line) > 0 and line[0] == '[':
                break
            if len(line) > 0:
                comments += line + "\n"
            count += 1
        dvs_lines = dvs_lines[count:]

        coordinate_system = re.findall(r'=\s*(.*)', dvs_lines[1])[0]
        normalisation = re.findall(r'=\s*(.*)', dvs_lines[2])[0]
        table = []

        for vector_def in dvs_lines[3:]:
            vector = re.findall(r'\(\s*([-+]?\d*\.\d+|\d+)\,\s*([-+]?\d*\.\d+|\d+)\,\s*([-+]?\d*\.\d+|\d+)\s*\)',
                                vector_def)

            if vector and len(vector) == 1:
                vector = vector[0]
                table.append([float(v) for v in vector])
        table = np.array(table)
        return DVSDirectionTable(table, comments, coordinate_system, normalisation)

    def _consume_table(self, dvs_lines):
        table_lines = []
        in_table = False
        for line in dvs_lines:
            if in_table and (len(line) > 0 and (line[0] == '[' or line[0] == '#')):
                break
            if len(line) > 0 and line[0] == '[':
                in_table = True
            table_lines.append(line)
        return self._parse_table(table_lines), dvs_lines[len(table_lines):]

    def _consume_top_comments(self, dvs_lines):
        comment_lines = []
        for line in dvs_lines:
            if len(line) > 0 and line[0] == '[':
                break
            comment_lines.append(line)
        return "\n".join(comment_lines), dvs_lines[len(comment_lines):]

    def _clean_lines(self, dvs_lines):
        dvs_lines = self._remove_windows_line_endings(dvs_lines)
        dvs_lines = self._trim_lines(dvs_lines)
        return dvs_lines

    def _trim_lines(self, dvs_lines):
        return [l.strip() for l in dvs_lines]

    def _remove_windows_line_endings(self, dvs_lines):
        return [l.replace('\r', '') for l in dvs_lines]


def read_dvs(file_name):
    """Read a DVS file from file.

    Args:
        file_name (str): The filename to read from

    Returns:
        DVS: A DVS object representation from the given file.
    """
    with open(file_name, 'r') as f:
        dvs_str = f.read()
        parser = DVSParser()
        return parser.parse(dvs_str)


def write_dvs(file_name, dvs):
    """Write the given DVS to the indicated file.

    Args:
        file_name (str): The filename to write to
        dvs (DVS): The dvs object to write
    """
    with open(file_name, 'w') as f:
        f.write(dvs.get_file_string())