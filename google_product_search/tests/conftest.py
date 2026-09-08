import asyncio
import inspect
import pytest

def pytest_pyfunc_call(pyfuncitem):
    """Run async test functions directly without pytest-asyncio plugin."""
    if inspect.iscoroutinefunction(pyfuncitem.function):
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(pyfuncitem.function(**pyfuncitem.funcargs))
        finally:
            loop.close()
            asyncio.set_event_loop(None)
        return True
    return None

@pytest.fixture(scope="function")
def event_loop():
    """Provide a fresh event loop for each test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
