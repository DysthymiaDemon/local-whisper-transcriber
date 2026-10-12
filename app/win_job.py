"""Windows Job Object limits for helper processes that handle untrusted input."""
from __future__ import annotations

import ctypes
import sys
from typing import Any

JobObjectExtendedLimitInformation = 9
JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
    )]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def contain_process(process: Any, memory_limit_bytes: int, kernel32: Any | None = None) -> int | None:
    """Put a started process in a job: memory cap, no child processes, killed when the job handle closes.

    Returns the job handle (keep it until the process is done, then pass it to close_job), or None when
    the limits could not be applied, for example outside Windows.
    """
    process_handle = getattr(process, "_handle", None)
    if process_handle is None or (sys.platform != "win32" and kernel32 is None):
        return None
    kernel32 = kernel32 or ctypes.WinDLL("kernel32", use_last_error=True)
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = (
        JOB_OBJECT_LIMIT_ACTIVE_PROCESS
        | JOB_OBJECT_LIMIT_PROCESS_MEMORY
        | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
        | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    )
    info.BasicLimitInformation.ActiveProcessLimit = 1
    info.ProcessMemoryLimit = int(memory_limit_bytes)
    configured = kernel32.SetInformationJobObject(
        job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)
    )
    if not configured or not kernel32.AssignProcessToJobObject(job, int(process_handle)):
        kernel32.CloseHandle(job)
        return None
    return job


def close_job(job: int | None, kernel32: Any | None = None) -> None:
    """Close the job handle; with KILL_ON_JOB_CLOSE this also ends any process still inside it."""
    if not job:
        return
    try:
        kernel32 = kernel32 or ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CloseHandle(job)
    except Exception:
        pass
