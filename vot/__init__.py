""" Some basic functions and classes used by the toolkit. """

import os
import logging

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

from attributee import Attributee, Integer, Boolean

class GlobalConfiguration(Attributee):
    """Global configuration object for the toolkit. It is used to store global configuration options.
    """

    debug_mode = Boolean(default=False, description="Enables debug mode for the toolkit.")
    sequence_cache_size = Integer(default=1000, description="Maximum number of sequences to keep in cache.")
    results_binary = Boolean(default=True, description="Enables binary results format.")
    mask_optimize_read = Boolean(default=True, description="Enables mask optimization when reading masks.")

    def __init__(self):
        """Initializes the global configuration object. It reads the configuration from environment variables.
        
        Raises:
            ValueError: When an invalid value is provided for an attribute.
        """
        kwargs = {}
        for k in self.attributes():
            envname = "VOT_{}".format(k.upper())
            if envname in os.environ:
                kwargs[k] = os.environ[envname]
        super().__init__(**kwargs)
        _logger.debug("Global configuration: %s", self)

    def __repr__(self):
        """Returns a string representation of the global configuration object."""
        return "debug_mode={} sequence_cache_size={} results_binary={} mask_optimize_read={}".format(
            self.debug_mode, self.sequence_cache_size, self.results_binary, self.mask_optimize_read
        )

config = GlobalConfiguration()

def check_debug() -> bool:
    """Checks if debug is enabled for the toolkit via an environment variable.

    Returns:
        bool: True if debug is enabled, False otherwise
    """
    return config.debug_mode

