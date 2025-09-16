from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any
from datetime import datetime

Role = Literal["user","system","assistant","tool","agent"]

class AgentEnvelope(BaseModel):
    trace_id: str                 # one user request across the system
    thread_id: str                # conversational thread (per chat session)
    from_agent: str               # e.g., "ba","doctor","nurse"
    to_agent: str                 # target agent or "orchestrator"
    role: Role = "agent"
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    at: datetime = Field(default_factory=datetime.utcnow)

class LogEvent(BaseModel):
    trace_id: str
    thread_id: str
    hop: Literal["request","response","note"]
    from_agent: str
    to_agent: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    at: datetime = Field(default_factory=datetime.utcnow)
