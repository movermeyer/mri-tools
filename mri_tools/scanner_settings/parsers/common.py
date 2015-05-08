__author__ = 'Robbert Harms'
__date__ = "2015-05-04"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ScannerSettingsParser(object):

    def get_value(self, settings_file, key):
        """Read a value for a given key for all sessions in the given file

        Args:
            settings_file (str): The settings file to read
            key (str): The name of the key to read

        Returns:
            dict: A dictionary with as keys the session names and as values the requested value for the key.
        """