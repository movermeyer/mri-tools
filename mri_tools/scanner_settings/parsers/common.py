__author__ = 'Robbert Harms'
__date__ = "2015-05-04"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ScannerSettingsParser(object):

    def get_value(self, key):
        """Read a value for a given key for all sessions in settings file.

        The settings file is supposed to be given in the constructor.

        Args:
            key (str): The name of the key to read

        Returns:
            dict: A dictionary with as keys the session names and as values the requested value for the key.
        """


class InfoReader(object):

    def get_read_out_time(self):
        """Get the read out time from the settings file.

        The settings file is supposed to be given to the constructor.

        Returns:
            float: The read out time in seconds.
        """