from pathlib import Path

import pytest
import skrf as rf

from skrf_usimmics import Substrate


@pytest.fixture(scope='session')
def data_dir():
    return Path(__file__).parent / 'data'


@pytest.fixture(scope='session')
def frequency():
    return rf.Frequency(0.1, 30, 401, 'GHz')


@pytest.fixture(scope='session')
def substrate():
    return Substrate(h=500e-6, ep_r=11.7)
