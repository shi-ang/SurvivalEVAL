"""Discrimination metrics using risk predictions that vary over time."""

from .time_dependent_concordance import concordance_time_dependent

__all__ = ["concordance_time_dependent"]
