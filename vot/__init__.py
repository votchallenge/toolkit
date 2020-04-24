
from .version import __version__

class VOTException(Exception):
    pass

def check_updates():
    import re
    import packaging
    import requests
    pattern = r"^__version__ = ['\"]([^'\"]*)['\"]"

    version_url = "https://github.com/votchallenge/vot-toolkit-python/raw/master/vot/version.py"

    try:
        response = requests.get(version_url, timeout=2)
    except:
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
