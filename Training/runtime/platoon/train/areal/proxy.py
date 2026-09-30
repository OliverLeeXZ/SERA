"""AReaL proxy session for tracking LLM interactions during rollouts."""

from __future__ import annotations

from areal.experimental.openai.proxy import (
    AReaLEndSessionRequest,
    ProxySession,
    _post_json_with_retry,
)


class ArealProxySession(ProxySession):
    """Async context manager for AReaL proxy sessions.

    This extends the base ProxySession to handle session lifecycle
    and ensure proper cleanup even on exceptions.
    """

    async def __aenter__(self) -> ArealProxySession:
        data = await _post_json_with_retry(
            self.http_session,
            f"{self.stripped_base_url}/rl/start_session",
            payload={},
        )
        self.session_id = data["session_id"]
        self.session_base_url = f"{self.stripped_base_url}/{self.session_id}"

        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        # Always try to end the session to prevent memory leaks in session_cache
        # even if an exception occurred during the rollout
        try:
            if self.session_id is not None:
                # On exception, set final_reward to 0 (failed rollout)
                reward = self.final_reward if exc_type is None else 0.0
                payload = AReaLEndSessionRequest(session_id=self.session_id, final_reward=reward).model_dump()
                await _post_json_with_retry(
                    self.http_session,
                    f"{self.stripped_base_url}/rl/end_session",
                    payload=payload,
                )
        except Exception as e:
            # Log but don't raise - we don't want to mask the original exception
            print(f"Warning: Failed to end session {self.session_id}: {e}")
        finally:
            await self.http_session.close()
