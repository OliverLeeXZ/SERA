import contextvars
import io
import re
import sys
from typing import Any, List, Optional

# -------------------------------------------------------------------------
# 1. Global proxy stream (installed exactly once at import time)
# -------------------------------------------------------------------------
_original_stdout = sys.stdout
_original_stderr = sys.stderr

_current_capture_out: contextvars.ContextVar[Optional[io.TextIOBase]] = contextvars.ContextVar(
    "current_capture_out", default=None
)
_current_capture_err: contextvars.ContextVar[Optional[io.TextIOBase]] = contextvars.ContextVar(
    "current_capture_err", default=None
)


class _Proxy(io.TextIOBase):
    def __init__(self, which: str, default_stream: io.TextIOBase) -> None:
        super().__init__()
        self._which = which
        self._default = default_stream

    def write(self, s: str) -> int:
        target = _current_capture_out.get() if self._which == "out" else _current_capture_err.get()
        (target or self._default).write(s)
        return len(s)

    def flush(self) -> None:
        target = (_current_capture_out.get() if self._which == "out" else _current_capture_err.get()) or self._default
        target.flush()

    def fileno(self) -> int:  # lets faulthandler work
        return self._default.fileno()

    def isatty(self) -> bool:  # nice for colorised logs
        return self._default.isatty()


# -------------------------------------------------------------------------
# Helper functions to (un)install proxy streams
# -------------------------------------------------------------------------
# We only want the proxy streams to be active while a ShellCapture context is
# running.  Installing them at import-time causes side-effects that can make a
# program appear to hang when an exception is raised in user code.  Instead we
# do a reference-counted, on-demand installation so that stdout/stderr revert
# back to their original values as soon as all ShellCapture scopes are closed.

_proxy_install_count = 0  # how many active ShellCapture contexts there are


def _install_proxies() -> None:
    """Install proxy streams if this is the first request."""
    global _proxy_install_count
    if _proxy_install_count == 0:
        sys.stdout = _Proxy("out", _original_stdout)
        sys.stderr = _Proxy("err", _original_stderr)
    _proxy_install_count += 1


def _uninstall_proxies() -> None:
    """Restore the original streams once the last capture has finished."""
    global _proxy_install_count
    if _proxy_install_count == 0:
        # This should never happen, but guard against negative counts.
        return
    _proxy_install_count -= 1
    if _proxy_install_count == 0:
        sys.stdout = _original_stdout
        sys.stderr = _original_stderr


# -------------------------------------------------------------------------
# 2. A minimal buffer stream for each shell
# -------------------------------------------------------------------------
class _Buffer(io.StringIO):
    """Just like StringIO but with a pop() helper."""

    def pop(self) -> str:
        """Read all data and clear the buffer (destructive)."""
        self.seek(0)
        data = self.read()
        self.truncate(0)
        self.seek(0)
        return data

    def peek(self) -> str:
        """Read all data without clearing the buffer (non-destructive)."""
        current_pos = self.tell()
        self.seek(0)
        data = self.read()
        self.seek(current_pos)
        return data

    def clear(self) -> None:
        """Clear the buffer."""
        self.truncate(0)
        self.seek(0)


# -------------------------------------------------------------------------
# 3. Display outputs buffer for rich content
# -------------------------------------------------------------------------
class _DisplayBuffer:
    """Buffer for capturing IPython display outputs."""

    def __init__(self) -> None:
        self.outputs: List[Any] = []

    def publish(self, data: Any, metadata: Optional[dict] = None) -> None:
        """Publish display data to the buffer."""
        self.outputs.append({"data": data, "metadata": metadata or {}})

    def pop(self) -> List[Any]:
        """Get all display outputs and clear the buffer."""
        outputs = self.outputs.copy()
        self.outputs.clear()
        return outputs

    def peek(self) -> List[Any]:
        """Get all display outputs without clearing the buffer."""
        return self.outputs.copy()

    def clear(self) -> None:
        """Clear all display outputs."""
        self.outputs.clear()


# -------------------------------------------------------------------------
# 4. Async context‑manager that sets the contextvars
# -------------------------------------------------------------------------
class ShellCapture:
    """
    Async-safe stdout/stderr/display capture using context variables.

    Usage:
        with ShellCapture() as cap:
            await shell.run_cell_async(code)
            print(cap.stdout())   # captured text
            print(cap.displays()) # captured rich outputs

    Async Safety:
    - ✅ Safe for typical sequential usage within async contexts
    - ✅ Uses contextvars for proper async task isolation
    - ⚠️  Don't share the same instance across concurrent tasks
    """

    def __init__(self, capture_display: bool = True) -> None:
        self._out_buf = _Buffer()
        self._err_buf = _Buffer()
        self._display_buf = _DisplayBuffer() if capture_display else None
        self._capture_display = capture_display

        # Store original hooks for restoration
        self._original_display_pub = None
        self._original_display_hook = None

    # getters -------------------------------------------------------------
    def stdout(self) -> str:
        """Get captured stdout without clearing the buffer."""
        return self._out_buf.peek()

    def stderr(self) -> str:
        """Get captured stderr without clearing the buffer."""
        return self._err_buf.peek()

    def displays(self) -> List[Any]:
        """Get captured display outputs without clearing the buffer."""
        return self._display_buf.peek() if self._display_buf else []

    def pop_stdout(self) -> str:
        """Get captured stdout and clear the buffer."""
        return self._out_buf.pop()

    def pop_stderr(self) -> str:
        """Get captured stderr and clear the buffer."""
        return self._err_buf.pop()

    def pop_displays(self) -> List[Any]:
        """Get captured display outputs and clear the buffer."""
        return self._display_buf.pop() if self._display_buf else []

    def clear(self) -> None:
        """Clear both stdout and stderr buffers."""
        self._out_buf.clear()
        self._err_buf.clear()
        if self._display_buf:
            self._display_buf.clear()

    # CM ------------------------------------------------------------
    def __enter__(self):
        # Swap context‑local targets; keep the tokens to restore later
        self._tok_out = _current_capture_out.set(self._out_buf)
        self._tok_err = _current_capture_err.set(self._err_buf)

        # Set up display capture if requested
        if self._capture_display:
            self._setup_display_capture()

        # Install proxies if this is the first capture context
        _install_proxies()

        return self

    def __exit__(self, exc_type, exc, tb):
        # Restore whatever was there before
        _current_capture_out.reset(self._tok_out)
        _current_capture_err.reset(self._tok_err)

        # Restore display hooks
        if self._capture_display:
            self._restore_display_capture()

        # Restore original streams
        _uninstall_proxies()

        # let exceptions propagate
        return False

    def _setup_display_capture(self) -> None:
        """Set up display publisher and display hook capture."""
        try:
            from IPython.core.displayhook import CapturingDisplayHook
            from IPython.core.displaypub import CapturingDisplayPublisher
            from IPython.core.getipython import get_ipython
        except ImportError:
            # IPython not available, skip display capture
            return

        shell = get_ipython()
        if shell is None:
            # Not running in IPython, skip display capture
            return

        # Capture display publisher
        self._original_display_pub = shell.display_pub
        shell.display_pub = CapturingDisplayPublisher()
        shell.display_pub.outputs = self._display_buf.outputs

        # Capture display hook (for last expression output)
        self._original_display_hook = sys.displayhook
        sys.displayhook = CapturingDisplayHook(shell=shell, outputs=self._display_buf.outputs)

    def _restore_display_capture(self) -> None:
        """Restore original display publisher and display hook."""
        try:
            from IPython.core.getipython import get_ipython
        except ImportError:
            return

        shell = get_ipython()
        if shell is None:
            return

        # Restore display publisher
        if self._original_display_pub is not None:
            shell.display_pub = self._original_display_pub

        # Restore display hook
        if self._original_display_hook is not None:
            sys.displayhook = self._original_display_hook


def strip_ansi_escape_sequences(text):
    """
    Remove ANSI escape sequences (color codes, formatting) from text.
    """
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_escape.sub("", text)
