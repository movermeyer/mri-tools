import re
import numpy as np

__author__ = 'Robbert Harms'
__date__ = "2015-04-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class DVS(object):

    def __init__(self, comments, dvs_direction_tables):
        """Create a new DVS object

        Args:
            comments (str): The string with the comments on top of the file
            dvs_direction_tables (list of DVSDirectionTable): The list with the direction tables

        Attributes:
            comments (str): The string with the comments on top of the file
            dvs_direction_tables (list of DVSDirectionTable): The list with the direction tables
        """
        self.comments = comments
        self.dvs_direction_tables = dvs_direction_tables

    def __repr__(self):
        """Get a complete string representation of the DVS. This can be written to file."""
        s = self.comments + "\n\n"
        for table in self.dvs_direction_tables:
            s += table.__repr__()
        return s


class DVSDirectionTable(object):

    def __init__(self, table, coordinate_system='xyz', normalisation='none'):
        """A representation of a direction table.

        Args:
            table (ndarray): The actual table
            coordinate_system (str): The coordinate system (for example 'xyz')
            normalisation (str): The normalisation definition (normally 'none')


        Attributes:
            table (ndarray): The actual table
            coordinate_system (str): The coordinate system (for example 'xyz')
            normalisation (str): The normalisation definition (normally 'none')
        """
        self.table = table
        self.coordinate_system = coordinate_system
        self.normalisation = normalisation

    def __repr__(self):
        """Get a complete string representation of this direction table. This can be written to file."""
        s = ''
        s += '[directions={}]'.format(self.table.shape[0]) + "\n"
        s += 'CoordinateSystem = {}'.format(self.coordinate_system) + "\n"
        s += 'Normalisation = {}'.format(self.normalisation) + "\n"
        for i in range(self.table.shape[0]):
            s += 'Vector[{0}] = ( {1}, {2}, {3} )'.format(i, *self.table[i, :]) + "\n"
        return s


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
        comments = self._get_comments(dvs_str)
        tables = self._parse_tables(dvs_str)
        return DVS(comments, tables)

    def _parse_tables(self, dvs_str):
        dvs_str = self._clean_string(dvs_str)

        table = np.eye(3)
        coordinate_system = 'xyz'
        normalisation = 'none'
        return [DVSDirectionTable(table, coordinate_system, normalisation)]

    def _get_comments(self, dvs_str):
        return [l for l in dvs_str if l[0] == '#']

    def _clean_string(self, dvs_str):
        s = self._remove_comments(dvs_str)
        s = self._remove_empty_lines(s)
        return s

    def _remove_comments(self, dvs_str):
        cleaner = re.compile('#.*')
        return cleaner.sub('', dvs_str)

    def _remove_empty_lines(self, dvs_str):
        return '\n'.join([s for s in dvs_str.split('\n') if not re.match(r'^\s*$', s)])


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
        f.write(dvs.__repr__())


if __name__ == '__main__':
    dvs = read_dvs('/home/robbert/Documents/phd/gradient_files/MBIC_DiffusionVectors.dvs')

    print(dvs)

    write_dvs('/home/robbert/Documents/phd/gradient_files/MBIC_DiffusionVectors_v2.dvs', dvs)