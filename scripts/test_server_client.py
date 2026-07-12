#!/usr/bin/env python
"""Small HTTP client for manually testing a running `sia serve` instance.

No dependencies beyond the standard library, so it runs in any Python 3.10+
environment regardless of which extras are installed -- it just needs the
server reachable over HTTP.

Usage:
    sia serve --port 8000                      # in one terminal
    python scripts/test_server_client.py        # in another

    python scripts/test_server_client.py "How do I reset my password?"
    python scripts/test_server_client.py --base-url http://127.0.0.1:9000 "..."
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

DEFAULT_TICKET = "I was charged twice for my subscription, please refund me."
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "expired"}


def _request(method: str, url: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def run_ticket(base_url: str, ticket: str, assistant_id: str, timeout: float) -> None:
    thread = _request("POST", f"{base_url}/threads", {})
    thread_id = thread["id"]
    print(f"thread:  {thread_id}")

    _request("POST", f"{base_url}/threads/{thread_id}/messages", {"role": "user", "content": ticket})
    print(f"ticket:  {ticket!r}")

    run = _request("POST", f"{base_url}/threads/{thread_id}/runs", {"assistant_id": assistant_id})
    run_id = run["id"]
    print(f"run:     {run_id} (assistant={assistant_id})")

    deadline = time.monotonic() + timeout
    while run["status"] not in TERMINAL_STATUSES and time.monotonic() < deadline:
        time.sleep(0.3)
        run = _request("GET", f"{base_url}/threads/{thread_id}/runs/{run_id}")

    print(f"status:  {run['status']}")
    if run["status"] != "completed":
        print(f"error:   {run.get('last_error')}")

    messages = _request("GET", f"{base_url}/threads/{thread_id}/messages")
    print("\n--- conversation ---")
    for message in reversed(messages["data"]):
        text = message["content"][0]["text"] if message["content"] else ""
        print(f"{message['role']}: {text}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ticket", nargs="?", default=DEFAULT_TICKET, help="Ticket text to send.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--assistant-id", default="classifier", help="Entry-point agent to run against.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Seconds to wait for the run to finish.")
    args = parser.parse_args()
    run_ticket(args.base_url, args.ticket, args.assistant_id, args.timeout)


if __name__ == "__main__":
    main()
