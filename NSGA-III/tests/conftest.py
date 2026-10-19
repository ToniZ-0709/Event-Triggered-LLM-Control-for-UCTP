"""Keep short behavioral CP tests separate from native-library cold-start time.

The CLI calibration workers still include their actual cold-start overhead.
"""
import pytest


@pytest.fixture(scope="session", autouse=True)
def ready_cp_runtime():
    from ortools.sat.python import cp_model
    return cp_model
