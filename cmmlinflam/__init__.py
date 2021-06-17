#
# Root of the cmmlinflam module.
# Provides access to all shared functionality.
#
# This file is part of CMMLINFLAM
# (https://github.com/I-Bouros/cmml-inflam.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#
"""cmmlinflam is a Epidemiology Modelling library.
It contains functionality for creating forward simulations .
"""

# Import version info
from .version_info import VERSION_INT, VERSION  # noqa

# Import main classes
from .stem import StemGillespie  # noqa
