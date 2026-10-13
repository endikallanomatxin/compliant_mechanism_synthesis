"""Disconnected clients must not obscure transport or application failures."""

import socket
import struct
import threading
import urllib.request
from http.server import HTTPServer

import pytest

from compliant_mechanism_synthesis.server import WorkbenchRequestHandler


def test_reset_client_does_not_log_exception_and_next_request_succeeds():
    entered = threading.Event()
    release = threading.Event()
    errors = []

    class Handler(WorkbenchRequestHandler):
        def do_GET(self):
            if self.path == "/abandon":
                entered.set()
                assert release.wait(5)
            body = b"healthy"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    class Server(HTTPServer):
        def handle_error(self, *args):
            errors.append(args)

    server = Server(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with socket.create_connection(server.server_address, timeout=5) as client:
            client.sendall(b"GET /abandon HTTP/1.0\r\n\r\n")
            assert entered.wait(5)
            client.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
            )
        release.set()
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.server_port}/healthy", timeout=5
        ) as response:
            assert response.read() == b"healthy"
        assert not errors
    finally:
        release.set()
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.mark.parametrize(
    "failure", [BrokenPipeError, ConnectionResetError, ConnectionAbortedError]
)
def test_expected_disconnect_closes_request(monkeypatch, failure):
    from http.server import BaseHTTPRequestHandler

    def disconnected(self):
        raise failure("Client disconnected")

    monkeypatch.setattr(BaseHTTPRequestHandler, "handle", disconnected)
    handler = object.__new__(WorkbenchRequestHandler)
    handler.handle()
    assert handler.close_connection


def test_application_errors_are_not_suppressed(monkeypatch):
    from http.server import BaseHTTPRequestHandler

    def broken(self):
        raise RuntimeError("Mechanics error")

    monkeypatch.setattr(BaseHTTPRequestHandler, "handle", broken)
    handler = object.__new__(WorkbenchRequestHandler)
    with pytest.raises(RuntimeError, match="Mechanics error"):
        handler.handle()
