"""Latency-probe tests use fake HTTPS connections, never credentials/network."""
import http.client
import json
import ssl
import unittest

from jev_isaac.jev import MAX_RESPONSE_BYTES, MOVES, SHOOTS
from jev_isaac.timing import run_timing


def response_body(model="jev-test"):
    def answer(choice, choices):
        return {"type": "choice", "choice": choice, "confidence": 1,
                "probabilities": {value: int(value == choice) for value in choices}}
    return json.dumps({"model": model, "answers": {
        "move": answer("up", MOVES), "shoot": answer("right", SHOOTS)},
        "usage": {"input_tokens": 700, "output_tokens": 80}}).encode()


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class Response:
    def __init__(self, body=None, status=200, will_close=False, failure=None):
        self.body = response_body() if body is None else body
        self.status = status
        self.will_close = will_close
        self.failure = failure
        self.bytes_read = 0
        self.read_calls = 0
        self.closed = False

    def read1(self, amount):
        self.read_calls += 1
        if self.failure is not None:
            raise self.failure
        chunk = self.body[self.bytes_read:self.bytes_read + amount]
        self.bytes_read += len(chunk)
        return chunk

    def close(self):
        self.closed = True


class Socket:
    def __init__(self):
        self.timeouts = []

    def settimeout(self, timeout):
        self.timeouts.append(timeout)


class Connection:
    def __init__(self, clock, responses, request_failure=None, request_time=0.04):
        self.clock = clock
        self.responses = iter(responses)
        self.request_failure = request_failure
        self.request_time = request_time
        self.sock = None
        self.connects = 0
        self.requests = []
        self.closed = False
        self.auto_open = 1

    def connect(self):
        self.connects += 1
        self.clock.advance(0.3)
        self.sock = Socket()

    def request(self, method, path, body, headers):
        self.requests.append((method, path, body, headers))
        self.clock.advance(self.request_time)
        if self.request_failure is not None:
            raise self.request_failure

    def getresponse(self):
        self.clock.advance(0.1)
        response = next(self.responses)
        if response.will_close:
            self.sock = None
        return response

    def close(self):
        self.closed = True
        self.sock = None


class TimingTests(unittest.TestCase):
    def test_saved_room_state_is_replayed_without_controls(self):
        from jev_isaac.demo import sample
        state = sample(321)
        state["player"]["x"] = 222
        result, connection, _ = self.invoke(samples=1, state=state)
        request = json.loads(connection.requests[0][2])
        self.assertEqual(request["state"]["frame"], 321)
        self.assertEqual(request["state"]["player"]["x"], 222)
        self.assertTrue(result["replayed_state"])
        self.assertFalse(result["synthetic_state"])
        self.assertFalse(result["game_controls_sent"])

    def test_response_diagnostic_report_has_safe_validation_code(self):
        body = json.loads(response_body())
        body["answers"]["shoot"]["choice"] = "untrusted-response-secret"
        result, connection, _ = self.invoke([Response(json.dumps(body).encode())])
        self.assertFalse(result["success"])
        self.assertEqual(result["validation_code"], "choice_unknown")
        self.assertEqual(result["question"], "shoot")
        self.assertEqual(result["requests"], 1)
        self.assertNotIn("untrusted-response-secret", json.dumps(result))

    def invoke(self, responses=None, **kwargs):
        clock = Clock()
        connection = Connection(clock, responses if responses is not None else [Response() for _ in range(4)],
                                kwargs.pop("request_failure", None), kwargs.pop("request_time", 0.04))
        factory_calls = []

        def factory(host, **options):
            factory_calls.append((host, options))
            return connection

        result = run_timing("offline-test-key", _connection_factory=factory, _clock=clock, **kwargs)
        return result, connection, factory_calls

    def test_one_verified_connection_reused_with_separate_connect_and_total_latency(self):
        result, connection, factory_calls = self.invoke()
        self.assertTrue(result["success"])
        self.assertEqual(result["requests"], 4)
        self.assertEqual(result["completed_samples"], 4)
        self.assertEqual(connection.connects, 1)
        self.assertEqual(connection.auto_open, 0)
        self.assertTrue(connection.closed)
        self.assertEqual(len(factory_calls), 1)
        host, options = factory_calls[0]
        self.assertEqual(host, "api.typesafe.ai")
        self.assertEqual(options["timeout"], 15)
        self.assertTrue(options["context"].check_hostname)
        self.assertEqual(options["context"].verify_mode, ssl.CERT_REQUIRED)
        self.assertEqual(result["initial_connect_ms"], 300)
        self.assertEqual([row["latency_ms"] for row in result["measurements"]], [440, 140, 140, 140])
        self.assertEqual([row["connection"] for row in result["measurements"]], ["new", "reused", "reused", "reused"])
        self.assertEqual(result["within_250ms"], 3)
        self.assertEqual(result["reused_within_250ms"], 3)
        self.assertEqual(result["reported_input_tokens"], 2800)
        self.assertEqual(result["reported_output_tokens"], 320)
        self.assertNotIn("offline-test-key", json.dumps(result))
        self.assertFalse(result["game_controls_sent"])
        for method, path, body, headers in connection.requests:
            self.assertEqual((method, path), ("POST", "/v1/systemone"))
            self.assertEqual(set(json.loads(body)["questions"]), {"move", "shoot"})

    def test_http_errors_and_redirects_stop_without_retry_or_reading_body(self):
        for status in (302, 401, 429, 503):
            response = Response(b"offline-test-key private server details", status)
            with self.subTest(status=status):
                result, connection, _ = self.invoke([response])
                self.assertFalse(result["success"])
                self.assertEqual(result["requests"], 1)
                self.assertEqual(result["completed_samples"], 0)
                self.assertIn(f"HTTP {status}", result["error"])
                self.assertNotIn("offline-test-key", json.dumps(result))
                self.assertEqual(response.read_calls, 0)
                self.assertTrue(response.closed)
                self.assertTrue(connection.closed)

    def test_read_limit_rejects_oversized_body_and_closes_connection(self):
        response = Response(b"x" * (MAX_RESPONSE_BYTES + 5000))
        result, connection, _ = self.invoke([response])
        self.assertFalse(result["success"])
        self.assertEqual(response.bytes_read, MAX_RESPONSE_BYTES + 1)
        self.assertEqual(result["requests"], 1)
        self.assertTrue(response.closed)
        self.assertTrue(connection.closed)

    def test_server_close_never_mislabeled_as_reuse_or_reconnected(self):
        result, connection, _ = self.invoke([Response(will_close=True)])
        self.assertFalse(result["success"])
        self.assertEqual(result["completed_samples"], 1)
        self.assertEqual(result["requests"], 1)
        self.assertEqual(connection.connects, 1)
        self.assertIn("no reconnect", result["error"])
        self.assertEqual(result["reported_input_tokens"], 700)
        self.assertEqual(result["reused_within_250ms"], 0)

    def test_close_on_final_requested_response_is_still_complete(self):
        result, connection, _ = self.invoke([Response(will_close=True)], samples=1)
        self.assertTrue(result["success"])
        self.assertEqual(result["requests"], 1)
        self.assertTrue(connection.closed)

    def test_transport_errors_are_redacted_and_not_retried(self):
        for failure in (OSError("offline-test-key"), TimeoutError("offline-test-key"),
                        http.client.BadStatusLine("offline-test-key")):
            with self.subTest(failure=type(failure)):
                result, connection, _ = self.invoke(request_failure=failure)
                self.assertFalse(result["success"])
                self.assertEqual(result["requests"], 1)
                self.assertNotIn("offline-test-key", json.dumps(result))
                self.assertTrue(connection.closed)

    def test_body_error_redacted_and_closed(self):
        response = Response(failure=http.client.IncompleteRead(b"offline-test-key"))
        result, connection, _ = self.invoke([response])
        self.assertFalse(result["success"])
        self.assertNotIn("offline-test-key", json.dumps(result))
        self.assertTrue(response.closed)
        self.assertTrue(connection.closed)

    def test_elapsed_budget_stops_before_next_network_stage(self):
        result, connection, _ = self.invoke(request_time=16)
        self.assertFalse(result["success"])
        self.assertEqual(result["requests"], 1)
        self.assertIn("timed out", result["error"])
        self.assertTrue(connection.closed)

    def test_caps_and_missing_credentials_fail_without_network(self):
        def no_connection(*args, **kwargs):
            self.fail("A connection must not be created")
        for count in (0, 5, -1, True, 1.5, "4"):
            with self.subTest(samples=count), self.assertRaises(ValueError):
                run_timing("offline-test-key", samples=count, _connection_factory=no_connection)
        for key in (None, "", "contains\ncontrol"):
            result = run_timing(key, _connection_factory=no_connection)
            self.assertFalse(result["success"])
            self.assertEqual(result["requests"], 0)

    def test_openrouter_fixed_endpoint_and_safe_reflected_model(self):
        result, connection, factory_calls = self.invoke([Response(response_body("offline-test-key"))],
                                                       samples=1, provider="openrouter")
        self.assertTrue(result["success"])
        self.assertEqual(factory_calls[0][0], "openrouter.ai")
        self.assertEqual(connection.requests[0][1], "/api/alpha/decisions")
        self.assertNotIn("offline-test-key", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
