from typing import Callable, Any
import asyncio

async def async_api_handler(func: Callable[..., Any], *args, **kwargs):
    try:
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    except Exception as e:
        return {"error": str(e), "status": 500}
