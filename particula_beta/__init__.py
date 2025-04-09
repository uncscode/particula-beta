"""
Particula-beta
"""
# pylint: disable=unused-import
# flake8: noqa
# pyright: basic

__version__ = "0.0.3"

from particula_beta import data
from particula_beta.time_manage import (
    time_str_to_epoch,
    relative_time,
    datetime64_from_epoch_array,
)
from particula_beta.units import convert_units