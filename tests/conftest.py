import asyncio

import pytest

from olol.script.oprotob import build


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def builder():
    """Fixture for the protobuf builder."""
    return build