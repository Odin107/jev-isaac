"""Synthetic game sender for exercising the real local transport without Isaac."""
from __future__ import annotations

import json
import socket
import threading
import time
from .controller import Controller, baseline


def sample(frame=30, enabled=True):
    return {"protocol":1, "type":"observation", "session":"synthetic-demo", "room_id":"demo-room-1",
            "frame":frame, "enabled":enabled, "paused":False,
            "player":{"x":320,"y":280,"vx":0,"vy":0,"hearts":6,"dead":False},
            "room":{"top_left":{"x":80,"y":140},"bottom_right":{"x":560,"y":420},"clear":False},
            "enemies":[{"x":400,"y":280,"vx":-1,"vy":0,"hp":10,"radius":12}],
            "projectiles":[],"hazards":[],"truncated":False}


def run_demo():
    ready = threading.Event()
    controller = Controller(baseline, port=0, max_calls=3, duration=2, logger=lambda _:None)
    result = {}
    def server():
        result.update(controller.run(ready))
    thread = threading.Thread(target=server, daemon=True)
    thread.start()
    if not ready.wait(2):
        raise RuntimeError("Demo listener failed to start")
    actions = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as game:
        game.settimeout(.02)
        for frame in range(30, 90, 3):
            game.sendto(json.dumps(sample(frame)).encode(), ("127.0.0.1", controller.bound_port))
            try:
                action = json.loads(game.recvfrom(60000)[0])
                if action["hold_frames"] > 1:
                    actions.append(action)
            except socket.timeout:
                pass
            if not thread.is_alive():
                break
            time.sleep(.02)
    thread.join(3)
    if thread.is_alive() or len(actions) < 1 or result.get("actions_sent") != 3:
        raise RuntimeError("Synthetic transport demo failed")
    return {"mode":"synthetic local demo", "real_game_tested":False, "jev_api_called":False,
            "result":"PASS", "controller":result, "example_action":actions[0]}
