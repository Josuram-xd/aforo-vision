"""CPU load controls: thread limits, process priority and the shared onnxruntime session options.

Why: several inference sessions (YOLO, OSNet, SCRFD, ArcFace) live in one process. By default each one keeps
a pool of spinning threads, so they fight for the cores even when idle (measured: ArcFace 52 ms -> ~2000 ms).
Spinning is therefore always off; `performance.cpu_threads` additionally caps every thread pool.
"""

import logging
import sys

logger = logging.getLogger(__name__)

_BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
_cpu_threads: int | None = None


def apply(cpu_threads: int | None, low_priority: bool, backend: str = "onnx") -> None:
    """Call once at startup, before the models are built (sessions read the limit when they are created)."""
    global _cpu_threads
    _cpu_threads = cpu_threads
    if low_priority:
        lower_process_priority()
    if cpu_threads:
        import cv2

        cv2.setNumThreads(cpu_threads)
        if backend == "torch":  # only import torch when it is going to be used anyway
            import torch

            torch.set_num_threads(cpu_threads)


def ort_session_options():
    """SessionOptions for every onnxruntime session we create: no idle spinning, optional thread cap."""
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    if _cpu_threads:
        options.intra_op_num_threads = _cpu_threads
    return options


def lower_process_priority() -> bool:
    """BELOW_NORMAL priority on Windows so the laptop stays responsive; no-op elsewhere. Never raises."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        ok = bool(kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), _BELOW_NORMAL_PRIORITY_CLASS))
    except Exception:  # noqa: BLE001 - priority is a nicety, never a reason to stop
        logger.debug("could not lower process priority", exc_info=True)
        return False
    logger.debug("process priority lowered: %s", ok)
    return ok
