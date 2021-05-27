import pytest
from market.exchange import Exchange


@pytest.fixture
def exchange():
    return Exchange()
