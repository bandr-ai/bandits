"""A fake OpenAI-compatible judge endpoint for tests. No real model is ever called."""

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeJudge:
    """OpenAI-compatible fake. Verdicts by answer text: WRONG -> all contradicted,
    HALF -> half supported, FLAKY -> alternates per call, else all supported."""

    def __init__(self):
        self.requests, self.mode, self.calls, self.rule = [], "ok", 0, None
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(
                    {"path": self.path, "auth": self.headers["Authorization"], **body}
                )
                outer.calls += 1
                if outer.mode == "500":
                    self.send_response(500)
                    self.end_headers()
                    return
                user = body["messages"][1]["content"]
                n = len(re.findall(r"^\d+\. ", user.split("CRITERIA:")[1], re.M))
                ans = re.search(r"<<<ANSWER\n(.*)\nANSWER>>>", user, re.S).group(1)
                if outer.rule:
                    v = outer.rule(ans, user, n)
                elif outer.mode == "yes":
                    v = ["supported"] * n
                elif outer.mode == "no":
                    v = ["contradicted"] * n
                elif "WRONG" in ans:
                    v = ["contradicted"] * n
                elif "HALF" in ans:
                    v = ["supported"] * (n // 2) + ["insufficient"] * (n - n // 2)
                elif "FLAKY" in ans and outer.calls % 2:
                    v = ["supported"] * (n - 1) + ["insufficient"]
                else:
                    v = ["supported"] * n
                content = json.dumps(
                    {"criteria": [{"id": i + 1, "verdict": x} for i, x in enumerate(v)]}
                )
                if outer.mode == "garbage":
                    content = "I think it is fine."
                out = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_port}/v1"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()
