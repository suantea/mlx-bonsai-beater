#!/usr/bin/env python3
"""mlx-router: 双机调度中心（统一入口 8080，主备自动路由）

- 主 (primary):   对面机器，经 SSH 隧道到达 127.0.0.1:8090
- 备 (backup):    本机服务 127.0.0.1:8081
- 每 2 秒健康检查主端；主端存活 → 全量转发主，否则自动切备。
- 客户端永远只连 http://127.0.0.1:8080 —— 局域网/出差切换对客户端透明。
"""
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.request

PRIMARY = os.environ.get("MLX_PRIMARY", "http://127.0.0.1:8090")
BACKUP = os.environ.get("MLX_BACKUP", "http://127.0.0.1:8081")
LISTEN = (os.environ.get("MLX_LISTEN", "127.0.0.1"), int(os.environ.get("MLX_PORT", "8080")))
CHECK_INTERVAL = float(os.environ.get("MLX_CHECK", "2"))

state = {"primary_ok": False}


def health_loop():
    """后台探活主端。主端 = SSH 隧道存活 ⇔ 对面服务可达。"""
    while True:
        ok = False
        try:
            with urllib.request.urlopen(f"{PRIMARY}/v1/models", timeout=2) as r:
                ok = r.status < 500
        except Exception:
            ok = False
        state["primary_ok"] = ok
        time.sleep(CHECK_INTERVAL)


def choose_upstream():
    return PRIMARY if state["primary_ok"] else BACKUP


class Router(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # 静默
        pass

    def _forward(self):
        upstream = choose_upstream()
        target = upstream + self.path
        body = None
        length = self.headers.get("Content-Length")
        if length:
            body = self.rfile.read(int(length))
        headers = {
            k: v
            for k, v in self.headers.items()
            if k.lower() not in ("host", "connection", "content-length")
        }

        req = urllib.request.Request(target, data=body, headers=headers, method=self.command)
        try:
            with urllib.request.urlopen(req, timeout=None) as resp:
                self.send_response(resp.status)
                keep = ("content-type", "date")
                for k, v in resp.headers.items():
                    if k.lower() in keep:
                        self.send_header(k, v)
                self.send_header("X-MLX-Route", "primary" if upstream == PRIMARY else "backup")
                self.end_headers()
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        break
        except Exception as e:
            self.send_error(502, f"upstream unavailable: {e}")

    def do_GET(self):
        self._forward()

    def do_POST(self):
        self._forward()


def main():
    t = threading.Thread(target=health_loop, daemon=True)
    t.start()
    srv = ThreadingHTTPServer(LISTEN, Router)
    print(f"[mlx-router] listening {LISTEN[0]}:{LISTEN[1]} primary={PRIMARY} backup={BACKUP}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()