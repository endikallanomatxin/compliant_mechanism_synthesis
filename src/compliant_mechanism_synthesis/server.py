"""Local web workbench with a serialized CPU mechanics backend."""

import argparse
import json
from dataclasses import dataclass, fields
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .nonlinear import ContinuationConfig
from .optimization import Search, SearchConfig


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000


class WorkbenchRequestHandler(BaseHTTPRequestHandler):
    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # Reloading or closing a tab can discard an in-flight FEM response.
            # Do not retry writing an error response to the disconnected socket.
            self.close_connection = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for config_type in (ServerConfig, SearchConfig, ContinuationConfig):
        defaults = config_type()
        for field in fields(defaults):
            value = getattr(defaults, field.name)
            parser.add_argument(
                "--" + field.name.replace("_", "-"), type=type(value), default=value
            )
    args = vars(parser.parse_args())
    build = lambda cls: cls(**{f.name: args[f.name] for f in fields(cls)})
    import torch

    torch.set_num_threads(1)
    search = Search(build(SearchConfig), build(ContinuationConfig))
    web = Path(__file__).parent / "web"

    class Handler(WorkbenchRequestHandler):
        def send_json(self, data, status=200):
            body = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.split("?")[0] == "/api/state":
                try:
                    from urllib.parse import parse_qs, urlsplit

                    index = int(
                        parse_qs(urlsplit(self.path).query).get("index", ["0"])[0]
                    )
                    self.send_json(search.snapshot(index))
                except (ValueError, RuntimeError) as exc:
                    self.send_json({"error": str(exc)}, 422)
            elif self.path in (
                "/",
                "/app.js",
                "/motion.js",
                "/configure.js",
                "/numerics.js",
                "/renderer.js",
                "/style.css",
            ):
                name = "index.html" if self.path == "/" else self.path[1:]
                body = (web / name).read_bytes()
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    {"html": "text/html", "js": "text/javascript", "css": "text/css"}[
                        name.rsplit(".", 1)[1]
                    ],
                )
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

        def do_POST(self):
            nonlocal search
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 65536:
                    raise ValueError("Request too large")
                data = json.loads(self.rfile.read(length))
                if self.path == "/api/reset":
                    dim = int(data["dimension"])
                    if dim not in (2, 3):
                        raise ValueError("Dimension must be 2 or 3")
                    from dataclasses import replace

                    search.writer.close()
                    search = Search(
                        replace(search.config, dimension=dim), search.continuation
                    )
                    self.send_json(search.snapshot())
                elif self.path in ("/api/specification", "/api/restart"):
                    from dataclasses import replace

                    from .mechanics import Design

                    restarting = self.path == "/api/restart"
                    settings = data.get("search", {}) if restarting else {}
                    continuation = data.get("continuation", {}) if restarting else {}
                    if restarting:
                        state = search.snapshot()
                        data = {
                            "design": state["specification"],
                            **state["waypoints"],
                        }
                    spec = data["design"]
                    positions = torch.tensor(spec["positions"], dtype=torch.float64)
                    if positions.ndim != 2 or not 4 <= len(positions) <= 32:
                        raise ValueError("Provide 4–32 nodes")
                    design = Design(
                        positions,
                        tuple(spec["roles"]),
                        torch.tensor(spec["edges"], dtype=torch.long),
                        positions.new_tensor(spec["radii"]),
                        positions.new_ones(len(spec["edges"])),
                    )
                    design.validate()
                    if sum(r == "mobile" for r in design.roles) < 2:
                        raise ValueError("Provide at least two mobile attachment nodes")
                    config = replace(
                        search.config,
                        dimension=design.dimension,
                        **settings,
                    )
                    candidate = Search(
                        config,
                        replace(search.continuation, **continuation),
                        design,
                    )
                    try:
                        candidate.set_trajectory(
                            data["factors"],
                            data["poses"],
                            float(data.get("rest_factor", 0)),
                        )
                        state = candidate.snapshot()
                        if not state["converged"]:
                            raise ValueError("The specified structure did not converge")
                    except (ValueError, RuntimeError):
                        candidate.writer.close()
                        raise
                    search.writer.close()
                    search = candidate
                    self.send_json(state)
                elif self.path == "/api/search-settings":
                    search.tune(data.get("search", {}), data.get("continuation"))
                    self.send_json(search.snapshot())
                elif self.path == "/api/step":
                    self.send_json(search.step(data["method"]))
                elif self.path == "/api/path-resolution":
                    from .resolution import compare_resolution

                    self.send_json(
                        compare_resolution(search, int(data.get("index", 0)))
                    )
                elif self.path == "/api/trajectory":
                    search.set_trajectory(
                        data["factors"],
                        data["poses"],
                        float(data.get("rest_factor", 0)),
                    )
                    self.send_json(search.snapshot())
                else:
                    self.send_error(404)
            except (ValueError, RuntimeError, KeyError, TypeError) as exc:
                self.send_json({"error": str(exc)}, 422)

    server_config = build(ServerConfig)
    server = HTTPServer((server_config.host, server_config.port), Handler)
    print(f"Workbench: http://{server_config.host}:{server_config.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        search.writer.close()
