"""Runtime network lockdown.

Internet access is only allowed while the installer downloads packages and models.
Once the app is running, everything must stay on this laptop. This module forces the
Hugging Face offline switches and blocks Python-level sockets (TCP connects, DNS lookups and
UDP sends) to non-loopback hosts.

Limits: these are Python hooks. They do not cover native libraries that open their own sockets,
other processes, or code that deliberately removes them. FFmpeg decoding is restricted separately
(local files only, in a contained child process). A hard guarantee needs an operating-system
control such as an outbound Windows Firewall rule for the runtime.
"""
from __future__ import annotations

import ipaddress
import os
import socket
import sys
from typing import Any

OFFLINE_ENV_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")
LOOPBACK_HOSTS = {"localhost", "localhost.localdomain", ""}

_guard_installed = False
_original_connect = socket.socket.connect
_original_connect_ex = socket.socket.connect_ex
_original_create_connection = socket.create_connection
_original_getaddrinfo = socket.getaddrinfo
_original_sendto = socket.socket.sendto
_original_sendmsg = getattr(socket.socket, "sendmsg", None)


class OfflineViolationError(ConnectionError):
    """Raised when runtime code tries to reach the network."""


def is_loopback_host(host: Any) -> bool:
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", errors="replace")
    text = str(host).strip().strip("[]").lower()
    if text in LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(text.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _check_address(address: Any) -> None:
    # AF_UNIX addresses are strings or bytes paths and never leave the machine.
    if not isinstance(address, tuple) or not address:
        return
    host = address[0]
    if not is_loopback_host(host):
        raise OfflineViolationError(f"Network access is disabled after setup (blocked connection to {host}).")


def _guarded_connect(self: socket.socket, address: Any) -> Any:
    _check_address(address)
    return _original_connect(self, address)


def _guarded_connect_ex(self: socket.socket, address: Any) -> Any:
    _check_address(address)
    return _original_connect_ex(self, address)


def _guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:
    _check_address(address)
    return _original_create_connection(address, *args, **kwargs)


def _guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
    if not is_loopback_host(host):
        raise OfflineViolationError(f"Network access is disabled after setup (blocked lookup of {host}).")
    return _original_getaddrinfo(host, *args, **kwargs)


def _guarded_sendto(self: socket.socket, data: Any, *args: Any) -> Any:
    # sendto(data, address) or sendto(data, flags, address)
    if args:
        _check_address(args[-1])
    return _original_sendto(self, data, *args)


def _guarded_sendmsg(self: socket.socket, buffers: Any, *args: Any) -> Any:
    # sendmsg(buffers[, ancdata[, flags[, address]]])
    if len(args) >= 3:
        _check_address(args[2])
    return _original_sendmsg(self, buffers, *args)


def force_offline_env() -> None:
    for name in OFFLINE_ENV_VARS:
        os.environ[name] = "1"
    # huggingface_hub reads HF_HUB_OFFLINE once at import. When the installer imported it
    # with downloads enabled, flip the cached flag as well.
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None:
        try:
            constants.HF_HUB_OFFLINE = True
        except Exception:
            pass


def install_socket_guard() -> None:
    global _guard_installed
    if _guard_installed:
        return
    socket.socket.connect = _guarded_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = _guarded_connect_ex  # type: ignore[method-assign]
    socket.create_connection = _guarded_create_connection  # type: ignore[assignment]
    socket.getaddrinfo = _guarded_getaddrinfo  # type: ignore[assignment]
    socket.socket.sendto = _guarded_sendto  # type: ignore[method-assign]
    if _original_sendmsg is not None:
        socket.socket.sendmsg = _guarded_sendmsg  # type: ignore[method-assign]
    _guard_installed = True


def remove_socket_guard() -> None:
    """Restore the original socket functions. Used by tests only."""
    global _guard_installed
    socket.socket.connect = _original_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = _original_connect_ex  # type: ignore[method-assign]
    socket.create_connection = _original_create_connection  # type: ignore[assignment]
    socket.getaddrinfo = _original_getaddrinfo  # type: ignore[assignment]
    socket.socket.sendto = _original_sendto  # type: ignore[method-assign]
    if _original_sendmsg is not None:
        socket.socket.sendmsg = _original_sendmsg  # type: ignore[method-assign]
    _guard_installed = False


def socket_guard_installed() -> bool:
    return _guard_installed


def enforce_offline() -> None:
    force_offline_env()
    install_socket_guard()
