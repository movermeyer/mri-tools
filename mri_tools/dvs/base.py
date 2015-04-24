__author__ = 'Robbert Harms'
__date__ = "2015-04-23"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class DVS(object):

    def __init__(self, comments, dvs_tables):
        """Create a new DVS object

        Args:
            comments (str): The list with comments on top of the file
            dvs_tables (list of DVSDirectionTable): The list with the direction tables

        Attributes:
            comments (str): The list with comments on top of the file
            dvs_tables (list of DVSDirectionTable): The list with the direction tables
        """
        self.comments = comments
        self.dvs_tables = dvs_tables

    def get_file_string(self, windows_line_endings=True):
        """Get a complete string representation of the DVS.

        Args:
            windows_line_endings (boolean): If we want to include an \r before every \n
        """
        s = self.comments + "\n"
        s += "\n".join([table.get_file_string(windows_line_endings=False) for table in self.dvs_tables])
        if windows_line_endings:
            s = s.replace("\n", "\r\n")
        return s

    def get_overview_representation(self):
        """Get a small overview of the contained contents."""
        s = 'Nmr tables: {}'.format(len(self.dvs_tables)) + "\n"
        for i, table in enumerate(self.dvs_tables):
            s += 'Table {}: {} directions'.format(i, table.table.shape[0]) + "\n"
        return s


class DVSDirectionTable(object):

    def __init__(self, table, comments='', coordinate_system='xyz', normalisation='none'):
        """A representation of a direction table.

        Args:
            table (ndarray): The actual table
            comments (str): The list with comments above this table
            coordinate_system (str): The coordinate system (for example 'xyz')
            normalisation (str): The normalisation definition (normally 'none')

        Attributes:
            table (ndarray): The actual table
            comments (str): The list with comments above this table
            coordinate_system (str): The coordinate system (for example 'xyz')
            normalisation (str): The normalisation definition (normally 'none')
        """
        self.table = table
        self.comments = comments
        self.coordinate_system = coordinate_system
        self.normalisation = normalisation

    def get_file_string(self, windows_line_endings=True):
        """Get a complete string representation of this direction table.

        Args:
            windows_line_endings (boolean): If we want to include an \r before every \n
        """
        s = self.comments
        s += '[directions={}]'.format(self.table.shape[0]) + "\n"
        s += 'CoordinateSystem = {}'.format(self.coordinate_system) + "\n"
        s += 'Normalisation = {}'.format(self.normalisation) + "\n"
        for i in range(self.table.shape[0]):
            s += 'Vector[{0}] = ( {1}, {2}, {3} )'.format(i, *self.table[i, :]) + "\n"
        if windows_line_endings:
            s = s.replace("\n", "\r\n")
        return s