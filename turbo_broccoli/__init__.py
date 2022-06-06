"""
.. include:: ../README.md
.. include:: ../CHANGELOG.md
"""
__docformat__ = "google"

from pkg_resources import get_distribution, DistributionNotFound

from turbo_broccoli.turbo_broccoli import (
    TurboBroccoliDecoder,
    TurboBroccoliEncoder,
)

try:
    __version__ = get_distribution("turbo-broccoli").version
except DistributionNotFound:
    __version__ = "local"
