import httpx
from .schemas import LogEvent

class LogClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def log(self, event: LogEvent) -> None:
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                await c.post(f"{self.base_url}/events", json=event.model_dump())
        except Exception:
            # Keep agents resilient even if logging is down
            pass
