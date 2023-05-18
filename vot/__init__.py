
import os
import logging
import typing

from .version import __version__

_logger = logging.getLogger("vot")

def get_logger() -> logging.Logger:
    """Returns the default logger object used to log different messages.

    Returns:
        logging.Logger: Logger handle
    """
    return _logger

class ToolkitException(Exception):
    """Base class for all toolkit related exceptions
    """
    pass


def toolkit_version() -> str:
    """Returns toolkit version as a string

    Returns:
        str: Version of the toolkit
    """
    return __version__

def check_updates() -> bool:
    """Checks for toolkit updates on Github, requires internet access, fails silently on errors.

    Returns:
        bool: True if an update is available, False otherwise.
    """

    import re
    import packaging.version as packaging
    import requests
    pattern = r"^__version__ = ['\"]([^'\"]*)['\"]"

    version_url = "https://github.com/votchallenge/vot-toolkit-python/raw/master/vot/version.py"

    try:
        get_logger().debug("Checking for new version")
        response = requests.get(version_url, timeout=5, allow_redirects=True)
    except Exception as e:
        get_logger().debug("Unable to retrieve version information %s", e)
        return False, None

    if not response:
        return False, None

    groups = re.search(pattern, response.content.decode("utf-8"), re.M)
    if groups:
        remote_version = packaging.parse(groups.group(1))
        local_version = packaging.parse(__version__)

        return remote_version > local_version, groups.group(1)

    else:
        return False, None

def _global_property(name: str, default: typing.Any = None, cast = str) -> typing.Any:

    var = cast(os.environ.get(name, default))

    return var

debug_mode = _global_property("VOT_TOOLKIT_DEBUG", False, lambda x: str(x).lower() in ["true", "1"])
sequence_cache_size = _global_property("VOT_SEQUENCE_CACHE", 1000, int)
results_binary = _global_property("VOT_RESULTS_BINARY", True, lambda x: str(x).lower() in ["true", "1"])

def check_debug() -> bool:
    """Checks if debug is enabled for the toolkit via an environment variable.

    Returns:
        bool: True if debug is enabled, False otherwise
    """
    return debug_mode

