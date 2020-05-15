
import os
import logging

from .version import __version__

class VOTException(Exception):
    pass

def check_updates():
    import re
    import packaging
    import requests
    pattern = r"^__version__ = ['\"]([^'\"]*)['\"]"

    version_url = "https://github.com/votchallenge/vot-toolkit-python/raw/master/vot/version.py"

    logger = logging.getLogger("vot")

    try:
        logger.debug("Checking for new version")
        response = requests.get(version_url, timeout=2)
    except Exception as e:
        logger.debug("Unable to retrieve version information %s", e)
        return False, None

    if not response:
        return False, None

    groups = re.search(pattern, response.raw, re.M)
    if groups:
        remote_version = packaging.version.parse(groups.group(1))
        local_version = packaging.version.parse(__version__)

        return remote_version > local_version, groups.group(1)

    else:
        return False, None

def check_debug():
    var = os.environ.get("VOT_TOOLKIT_DEBUG", "false").lower()
    return var in ["true", "1"]

