""" Some basic functions and classes used by the toolkit. """

import os
import logging

from .version import __version__

from lazy_object_proxy import Proxy

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
    """Global configuration object for the toolkit. It is used to store global configuration options. It can be initialized
    from environment variables. The following options are supported:

    - ``VOT_DEBUG_MODE``: Enables debug mode for the toolkit.
    - ``VOT_SEQUENCE_CACHE_SIZE``: Maximum number of sequences to keep in cache.
    - ``VOT_RESULTS_BINARY``: Enables binary results format.
    - ``VOT_MASK_OPTIMIZE_READ``: Enables mask optimization when reading masks.
    - ``VOT_WORKER_POOL_SIZE``: Number of workers to use for parallel processing.
    - ``VOT_PERSISTENT_CACHE``: Enables persistent cache for analysis results in workspace.

    """

    debug_mode = Boolean(default=False, description="Enables debug mode for the toolkit.")
    sequence_cache_size = Integer(default=100, description="Maximum number of sequences to keep in cache.")
    results_binary = Boolean(default=True, description="Enables binary results format.")
    mask_optimize_read = Boolean(default=True, description="Enables mask optimization when reading masks.")
    worker_pool_size = Integer(default=1, description="Number of workers to use for parallel processing.")
    persistent_cache = Boolean(default=True, description="Enables persistent cache for analysis results in workspace.")

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

    def __repr__(self):
        """Returns a string representation of the global configuration object."""
        return " ".join(["{}={}".format(k, getattr(self, k)) for k in self.attributes()])

#_logger = None

from vot.utilities import singleton

@singleton
def get_logger() -> logging.Logger:
    """Returns the default logger object used to log different messages.

    Returns:
        logging.Logger: Logger handle
    """

    def init():
        from .utilities import ColoredFormatter
        logger = logging.getLogger("vot")
        stream = logging.StreamHandler()
        stream.setFormatter(ColoredFormatter())
        logger.addHandler(stream)
        if check_debug():
            logger.setLevel(logging.DEBUG)
        return logger

    return Proxy(init)

config = Proxy(lambda: GlobalConfiguration())

def check_debug() -> bool:
    """Checks if debug is enabled for the toolkit via an environment variable.

    Returns:
        bool: True if debug is enabled, False otherwise
    """
    return config.debug_mode

def print_config():
    """Prints the global configuration object to the logger."""
    if check_debug():
        get_logger().debug("Configuration: %s", config)
