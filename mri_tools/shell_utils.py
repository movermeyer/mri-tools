import os

__author__ = 'Robbert Harms'
__date__ = "2015-05-07"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def binary_in_path(command_name):
    """Check if the given command name for a binary exists in the users path.

    Args:
        command_name (str): the name of the command to check for existence and executability

    Returns:
        bool: true if the command can be found and is executable, false otherwise.
    """
    for path_dir in os.environ["PATH"].split(os.pathsep):
        full_path = os.path.join(path_dir.strip('"'), command_name)
        if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
            return True
    return False


def get_fsl_path():
    """Get the path to the FSL dir

    Returns:
        str: The path to the FSL directory.

    Raises:
        EnvironmentError: If the path could not be found.
    """
    fsl_path = os.environ.get('FSLDIR', '')
    if not fsl_path:
        raise EnvironmentError('The Environment variable FSLDIR is not set')


def get_fsl_command(application_name):
    """Get the correct command name for the given FSL application.

    Args:
        application_name (str): The name of the application we want the correct command of.

    Returns:
        str: Either the given application name or the name with fsl-5.0 prepended to it.

    Raises:
        EnvironmentError: if the fsl program could not be found
    """
    if binary_in_path(application_name):
        return application_name

    prefixed_name = 'fsl-5.0' + application_name
    if binary_in_path(prefixed_name):
        return prefixed_name

    raise EnvironmentError('Could not find FSL program {}'.format(application_name))