"""Discrimination metrics using risk predictions that vary over time."""

from .TimeDependentConcordance import concordance_time_dependent

__all__ = ["concordance_time_dependent"]
