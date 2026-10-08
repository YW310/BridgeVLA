"""Serve a trained OHT checkpoint using the versioned absolute-TCP protocol."""
import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from .runtime.protocol import request_identity, action_response
from .runtime.transport import unpack_observation, MAX_BYTES
from .runtime.predicted_wrapper import policy_observation
from .data.observation import validate_observation


class Session:
    def __init__(self, policy):
        self.policy = policy
        self.episode, self.step, self.timestamp, self.goal = None, 0, -1, None
        self.seen = set()
        self.failed = False

    def act(self, request):
        identity = request_identity(request)
        if request.get("contract_sha256") != self.policy.contract["sha256"]:
            raise ValueError("Client/server replay contract mismatch")
        goal = request.get("goal")
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal must be a nonempty task instruction")
        new = identity["episode"] != self.episode
        if new:
            if identity["step"] != 0 or identity["episode"] in self.seen:
                raise ValueError("A new episode must use a fresh identifier and start at step zero")
        elif self.failed or identity["step"] != self.step or identity["timestamp"] <= self.timestamp or goal != self.goal:
            raise ValueError("Out-of-order/stale request or changed goal; failed episodes require a new ID")
        observation = unpack_observation(request["observation"])
        observation = policy_observation(observation, self.policy.contract["data_config"]["cameras"])
        validate_observation(observation, self.policy.contract["data_config"])
        if new:
            self.policy.reset()
            self.episode, self.step, self.timestamp, self.goal = identity["episode"], 0, -1, goal
            self.seen.add(self.episode)
            self.failed = False
        try:
            action = self.policy.act(observation, goal, identity["step"])
            response = action_response(request, action)
        except Exception:
            self.failed = True
            raise
        self.step, self.timestamp = identity["step"] + 1, identity["timestamp"]
        response["contract_sha256"] = self.policy.contract["sha256"]
        return response


def handler(session):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/health":
                self.send_error(404)
                return
            self.reply(200, dict(ready=True, schema="bridgevla_oht_absolute_tcp_v1",
                                 contract_sha256=session.policy.contract["sha256"]))

        def reply(self, status, value):
            body = json.dumps(value, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/act":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= MAX_BYTES:
                    raise ValueError("Missing or oversized Content-Length")
                value = json.loads(self.rfile.read(length))
                self.reply(200, session.act(value))
            except (ValueError, KeyError, TypeError) as exc:
                self.reply(400, dict(error=str(exc)))
            except Exception:
                logging.exception("OHT policy inference failed")
                self.reply(500, dict(error="Policy inference failed; reset with a new episode ID"))
    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pretrain-path")
    parser.add_argument("--predictor")
    parser.add_argument("--predictor-provenance")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args(argv)
    from .runtime.loading import load_policy
    policy = load_policy(args.checkpoint, args.device, args.pretrain_path,
                         args.predictor, args.predictor_provenance)
    server = HTTPServer((args.host, args.port), handler(Session(policy)))
    print(json.dumps(dict(ready=True, host=args.host, port=server.server_port,
                          contract_sha256=policy.contract["sha256"])), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
