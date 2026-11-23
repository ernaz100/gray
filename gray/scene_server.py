#!/usr/bin/env python3
"""Serve the Loc_08 viewer and pass its prompts to gray/scene_worker.py on Viper. Runs on the workstation.

    python3 gray/scene_server.py --static ~/cursor-workspace/loc08      # then open http://127.0.0.1:8771/viewer/egoprimo.html

The viewer page and the scene are the ones built for the earlier project (``--static`` is their
folder); the page works unchanged because the worker speaks the same protocol. Only the model
behind it differs: the head-path model and the body model of this repository.

POST /api/generate takes {"pos": [x, y, z], "fwd": [x, y, z], "text": "...", "seed": 0} and returns the
worker's reply. GET /api/status says whether the worker is up, GET /api/body returns the body mesh.

The worker is one long-running step inside the user's job named ``gpusession`` on Viper, started on the
first request with ``hpc gpu exec viper``. This script never allocates a GPU: start the session first, for example

    sbatch -J gpusession -p apudev --constraint=apu --gres=gpu:1 -c 24 --mem=110000 -t 15 --wrap "sleep infinity"

(15 minutes on the test partition; without -p and with a longer -t for a longer session.)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REMOTE = ("cd /u/enazaren/projects/gray && module load ffmpeg/7.1 2>/dev/null; export TORCH_BLAS_PREFER_HIPBLASLT=0; "
          "exec /u/enazaren/envs/torch/bin/python -u -m gray.scene_worker")


class Worker:
    def __init__(self, cluster: str) -> None:
        self.cluster, self.lock = cluster, threading.Lock()
        self.process: subprocess.Popen | None = None
        self.info: dict = {}
        self.body: dict | None = None
        self.next_id = 0

    def _read(self) -> dict:
        """The next reply line; other output of the remote step is skipped."""
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("the worker stopped; see the server's terminal for its messages (is a gpusession running?)")
            if line.startswith("@@"):
                return json.loads(line[2:])

    def ask(self, request: dict) -> dict:
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                print(f"starting the worker in the gpusession on {self.cluster}...", flush=True)
                self.process = subprocess.Popen(["hpc", "gpu", "exec", self.cluster, "bash", "-lc", REMOTE],
                                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr, text=True, bufsize=1)
                self.info = self._read()
                print(f"worker ready: {self.info}", flush=True)
            self.next_id += 1
            request["id"] = self.next_id
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            while True:
                answer = self._read()
                if answer.get("id") == self.next_id:
                    return answer

    def mesh(self) -> dict:
        if self.body is None:
            answer = self.ask({"body": True})
            if "error" in answer:
                return answer
            self.body = answer
        return self.body

    def status(self) -> dict:
        running = self.process is not None and self.process.poll() is None
        return {"running": running, **(self.info if running else {})}

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.stdin.close()
            self.process.terminate()


class Handler(SimpleHTTPRequestHandler):
    worker: Worker

    def _json(self, payload: dict, code: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/api/status":
            self._json(self.worker.status())
        elif self.path == "/api/body":
            try:
                self._json(self.worker.mesh())
            except Exception as error:  # noqa: BLE001
                self._json({"error": str(error)}, 500)
        else:
            super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/generate":
            self._json({"error": "unknown path"}, 404)
            return
        try:
            request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            self._json(self.worker.ask({key: request[key] for key in ("pos", "fwd", "text", "seed") if key in request}))
        except Exception as error:  # noqa: BLE001
            self._json({"error": str(error)}, 500)

    def log_message(self, pattern: str, *args) -> None:
        if args and "/api/" in str(args[0]):
            super().log_message(pattern, *args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--static", type=Path, required=True, help="the folder with viewer/egoprimo.html and the scene")
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--cluster", default="viper")
    args = parser.parse_args()
    Handler.worker = Worker(args.cluster)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(args.static.expanduser())))
    print(f"http://127.0.0.1:{args.port}/viewer/egoprimo.html", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        Handler.worker.stop()


if __name__ == "__main__":
    main()
