from contextvars import ContextVar

from .env import CodeActEnv as CodeActEnv
from .env import IPythonCodeExecutor as IPythonCodeExecutor
from .env import SafeAsyncio as SafeAsyncio
from .env import safe_asyncio as safe_asyncio
from .types import CodeActAction as CodeActAction
from .types import CodeActObservation as CodeActObservation
from .types import CodeActStep as CodeActStep
from .types import CodeExecutor as CodeExecutor
from .types import ForkableCodeExecutor as ForkableCodeExecutor

executor_context: ContextVar[dict] = ContextVar("code_executor_context")
