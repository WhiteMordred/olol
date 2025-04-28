import unittest.mock as mock

import pytest

from src.ollama_async_client import AsyncOllamaClient
from src.ollama_async_server import AsyncOllamaService


@pytest.fixture
async def service():
    return AsyncOllamaService()

@pytest.fixture
async def client():
    client = AsyncOllamaClient()
    yield client
    await client.close()

@pytest.mark.asyncio
async def test_run_model(service, client):
    with mock.patch('asyncio.create_subprocess_exec') as mock_exec:
        # Setup mock
        mock_process = mock.AsyncMock()
        mock_process.communicate.return_value = (b"Test response", b"")
        mock_process.returncode = 0
        mock_exec.return_value = mock_process

        response = await client.run_model("test-model", "test prompt")
        assert response == "Test response"

@pytest.mark.asyncio
async def test_create_session(service, client):
    session_id = "test-session"
    success = await client.create_session(session_id, "test-model")
    assert success == True
    assert session_id in service.active_sessions