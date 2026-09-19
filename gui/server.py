#!/usr/bin/env python3
"""mlx-bonsai-beater GUI：本地兜底聊天窗口（oMLX / llama.cpp webui 风格）。

- 反向代理到调度中心（默认 http://127.0.0.1:8080，可用 MLX_UPSTREAM 覆盖）
- 流式转发 SSE；并在响应头回传 X-MLX-Route（primary/backup）供前端显示主备状态
- 纯 Python 标准库，无任何第三方依赖

用法:
  python3 gui/server.py [--port 7860] [--upstream http://127.0.0.1:8080]
"""
import argparse
import os
import socket
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import posixpath

ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(ROOT, "static")
UPSTREAM = os.environ.get("MLX_UPSTREAM", "http://127.0.0.1:8080")


def resolve_ipv4(url):
    """把 .local（mDNS）等主机名解析成 IPv4 地址，避免 urllib 走 IPv6 link-local。"""
    from urllib.parse import urlsplit, urlunsplit
    parts = urlsplit(url)
    if not parts.hostname:
        return url
    try:
        ips = socket.getaddrinfo(parts.hostname, None, socket.AF_INET)
    except OSError:
        return url
    if not ips:
        return url
    ip = ips[0][4][0]
    port = parts.port
    netloc = ip if port is None else f"{ip}:{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    # ---- upstream proxy -------------------------------------------------
    def _proxy(self):
        target = UPSTREAM + self.path
        body = None
        length = self.headers.get("Content-Length")
        if length:
            body = self.rfile.read(int(length))
        headers = {
            k: v
            for k, v in self.headers.items()
            if k.lower() not in ("host", "connection", "content-length", "accept-encoding")
        }
        headers["Accept-Encoding"] = "identity"
        req = urllib.request.Request(target, data=body, headers=headers, method=self.command)
        try:
            with urllib.request.urlopen(req, timeout=None) as resp:
                self.send_response(resp.status)
                keep = ("content-type",)
                for k, v in resp.headers.items():
                    if k.lower() in keep:
                        self.send_header(k, v)
                route = resp.headers.get("X-MLX-Route")
                if route:
                    self.send_header("X-Proxied-Route", route)
                self.send_header("Cache-Control", "no-store")
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

    # ---- static files ---------------------------------------------------
    def _static(self):
        path = posixpath.normpath("/" + self.path)
        parts = path.lstrip("/").split("/")
        if not parts or parts[0] == "":
            parts = ["index.html"]
        safe = os.path.join(STATIC, *parts)
        if not os.path.abspath(safe).startswith(os.path.abspath(STATIC)) or not os.path.isfile(safe):
            self.send_error(404)
            return
        ctype = "text/html"
        if safe.endswith(".js"):
            ctype = "text/javascript"
        elif safe.endswith(".css"):
            ctype = "text/css"
        with open(safe, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _status(self):
        import json
        payload = json.dumps({"upstream": UPSTREAM, "ok": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/api/status":
            self._status()
        elif self.path.startswith("/v1/"):
            self._proxy()
        else:
            self._static()

    def do_POST(self):
        if self.path.startswith("/v1/"):
            self._proxy()
        else:
            self.send_error(404)


def main():
    global UPSTREAM
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("MLX_GUI_PORT", "7860")))
    ap.add_argument("--upstream", default=UPSTREAM)
    args = ap.parse_args()
    UPSTREAM = resolve_ipv4(args.upstream.rstrip("/"))
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[mlx-gui] http://127.0.0.1:{args.port}  upstream={UPSTREAM}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()