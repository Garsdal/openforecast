"""Shared native-parameter reflection used by the Darts integration."""

from openforecast.providers.native import (
    Parameter,
    ParameterDiscovery,
    checked,
    named,
    parameters_from_signature,
    schema_of,
)

__all__ = [
    "Parameter",
    "ParameterDiscovery",
    "checked",
    "named",
    "parameters_from_signature",
    "schema_of",
]
