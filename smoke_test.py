"""Stdio-level smoke test for the pg-mcp MCP server.

Speaks raw JSON-RPC over stdin/stdout (MCP protocol) to verify:
1. initialize handshake
2. tools/list
3. tools/call on the `query` tool (return_type=sql, no LLM key needed upstream)
"""

import json
import subprocess
import sys

PROJECT = r"E:/big-model-project/geektime-bootcamp-ai/w5/pg-mcp"
PY = PROJECT + "/.venv/Scripts/python.exe"


def send(proc, obj):
    proc.stdin.write((json.dumps(obj) + "\n").encode())
    proc.stdin.flush()


def recv(proc):
    # Skip non-JSON lines (pretty log output goes to stdout on Windows)
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("server closed stdout")
        text = line.decode("utf-8", errors="replace").strip()
        if not text.startswith(("{", "[")):
            continue
        return json.loads(text)


def main():
    proc = subprocess.Popen(
        [PY, "-X", "utf8", "-m", "pg_mcp"],
        cwd=PROJECT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        send(proc, {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "smoke", "version": "0.1"},
            },
        })
        init = recv(proc)
        print("1) initialize OK ->", init["result"]["serverInfo"]["name"])

        send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools = recv(proc)
        names = [t["name"] for t in tools["result"]["tools"]]
        print("2) tools/list OK ->", names)

        send(proc, {
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {
                "name": "query",
                "arguments": {
                    "question": "Count all users",
                    "return_type": "sql",
                },
            },
        })
        resp = recv(proc)
        content = resp["result"]["content"][0]["text"]
        print("3) tools/call OK ->", content[:300])
        print("\nSMOKE TEST PASSED")
    finally:
        proc.terminate()


if __name__ == "__main__":
    sys.exit(main())
