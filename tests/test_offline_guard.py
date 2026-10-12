import os
import socket
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import offline_guard
from offline_guard import OfflineViolationError, enforce_offline, is_loopback_host


class OfflineGuardTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(offline_guard.remove_socket_guard)

    def test_env_is_forced_even_when_user_disabled_offline_mode(self):
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"}):
            enforce_offline()
            for name in offline_guard.OFFLINE_ENV_VARS:
                self.assertEqual(os.environ[name], "1")

    def test_already_imported_huggingface_constants_are_flipped(self):
        constants = types.ModuleType("huggingface_hub.constants")
        constants.HF_HUB_OFFLINE = False
        with patch.dict(sys.modules, {"huggingface_hub.constants": constants}), patch.dict(os.environ, {}):
            enforce_offline()
        self.assertTrue(constants.HF_HUB_OFFLINE)

    def test_external_connections_and_lookups_are_blocked(self):
        with patch.dict(os.environ, {}):
            enforce_offline()
        with self.assertRaises(OfflineViolationError):
            socket.create_connection(("huggingface.co", 443), timeout=1)
        with self.assertRaises(OfflineViolationError):
            socket.getaddrinfo("huggingface.co", 443)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            with self.assertRaises(OfflineViolationError):
                sock.connect(("93.184.216.34", 80))

    def test_urllib_cannot_reach_the_internet(self):
        import urllib.request

        with patch.dict(os.environ, {}):
            enforce_offline()
        with self.assertRaises(Exception) as caught:
            urllib.request.urlopen("https://huggingface.co", timeout=1)
        self.assertIn("Network access is disabled", str(caught.exception))

    def test_loopback_stays_allowed(self):
        with patch.dict(os.environ, {}):
            enforce_offline()
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.addCleanup(server.close)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.create_connection(server.getsockname(), timeout=1):
            pass
        self.assertTrue(socket.getaddrinfo("localhost", 80))

    def test_guard_is_idempotent_and_removable(self):
        with patch.dict(os.environ, {}):
            enforce_offline()
            first = socket.create_connection
            enforce_offline()
        self.assertIs(socket.create_connection, first)
        offline_guard.remove_socket_guard()
        self.assertFalse(offline_guard.socket_guard_installed())
        self.assertIs(socket.create_connection, offline_guard._original_create_connection)

    def test_loopback_host_detection(self):
        for host in ("localhost", "127.0.0.1", "::1", "[::1]", None, b"127.0.0.5"):
            self.assertTrue(is_loopback_host(host), host)
        for host in ("huggingface.co", "8.8.8.8", "10.0.0.1"):
            self.assertFalse(is_loopback_host(host), host)


if __name__ == "__main__":
    unittest.main()
