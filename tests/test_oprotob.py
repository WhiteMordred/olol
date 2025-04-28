import pytest

from olol.script.oprotob import build


def test_build_exists():
    """Test that the build function exists."""
    assert callable(build)

@pytest.mark.asyncio
async def test_build_returns_success(builder):
    """Test that the build function returns success."""
    result = await builder()
    assert result == 0