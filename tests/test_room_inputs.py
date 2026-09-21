"""Activity and firing choices combine without reselecting movement or targets."""
import copy
from pathlib import Path
import sys
import unittest

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).resolve().parents[1]/"src")]
from test_adventure import ready as base_ready
from test_controller_arming import OLD, NEW
from test_player_controller import frame, run
from test_player_policy import call
from test_props import fire
from jev_isaac.player_navigation import PlayerNavigator
from jev_isaac.exploration import _validated
from jev_isaac.room_inputs import room_input_candidates, room_input_move


def ready():
    data = base_ready()
    data["player"].update(weapon_type=1, weapon_types=[1])
    return data


def choose_input(key, direction="none"):
    def choose(payload):
        result = {"fire": direction}
        if "activity" in payload["questions"]:
            if key == "wait":
                result["activity"] = "wait"
            else:
                selected = next(c for c in payload["state"]["activity_candidates"] if c["key"] == key)
                result["activity"] = selected["option"]
        return result
    return choose


class RoomInputTests(unittest.TestCase):
    def test_parallel_fire_choice_exists_for_every_observed_weapon(self):
        for weapon in range(1, 16):
            with self.subTest(weapon=weapon):
                data = ready()
                data["player"].update(weapon_type=weapon, weapon_types=[weapon])
                nav = PlayerNavigator()
                nav.step(data, 0)
                data["frame"] += 20
                nav.step(data, .7)
                data["_adventure_options"] = [c.as_dict() for c in nav.adventure_options]
                decision, request = call(data, {"activity": "wait", "fire": "down"})
                self.assertEqual(set(request["questions"]), {"activity", "fire"})
                self.assertEqual(set(request["questions"]["fire"]["criteria"]),
                                 {"none", "left", "right", "up", "down"})
                self.assertEqual(decision.fire_direction, "down")

    def test_technology_moves_and_fires_together_then_updates_fire_without_restarting_move(self):
        data = ready()
        data["player"].update(weapon_type=3, weapon_types=[3])
        data["hazards"] = [fire(x=400, y=280, hp=2)]
        def choose(payload):
            return choose_input("move_to:up", "right" if "activity" in payload["questions"] else "left")(payload)
        events = [(i/30, frame(data, 30+i), OLD) for i in range(48)]
        transport, requests, _, result = run(events, choose, duration=1.6)
        self.assertEqual(set(requests[0]["questions"]), {"activity", "fire"})
        self.assertTrue(any(set(r["questions"]) == {"fire"} for r in requests[1:]))
        for direction in ("left", "right"):
            self.assertTrue(any(p["move"] == "up" and p["shoot"] == direction for _, p, _ in transport.sent))
        selected = [e for e in result["adventure_progress"]["player_events"] if e["event"] == "selected"]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["key"], "move_to:up")
        self.assertEqual(data["hazards"][0]["hp"], 2)
        self.assertEqual(result["errors"], 0)

    def test_wait_can_fire_but_none_releases_input(self):
        for direction in ("right", "none"):
            with self.subTest(direction=direction):
                data = ready()
                events = [(i/30, frame(data, 30+i), OLD) for i in range(30)]
                transport, _, _, _ = run(events, choose_input("wait", direction), duration=1)
                after = [p for when, p, _ in transport.sent if .7 < when < .95]
                self.assertTrue(after)
                self.assertTrue(all(p["move"] == "none" and p["shoot"] == direction for p in after))

    def test_no_fresh_reply_expires_firing_without_restarting_movement(self):
        data = ready()
        events = [(i/30, frame(data, 30+i), OLD) for i in range(99)]
        transport, _, _, result = run(events, choose_input("enter:2:85", "right"),
            delay=lambda count: 0 if count == 1 else 3, duration=3.3)
        self.assertTrue(any(p["shoot"] == "right" for _, p, _ in transport.sent))
        after = [p for when, p, _ in transport.sent if 2.8 < when < 3.2]
        self.assertTrue(after)
        self.assertTrue(all(p["shoot"] == "none" for p in after))
        self.assertEqual(result["errors"], 0)

    def test_pause_and_rearm_discard_inflight_parallel_fire(self):
        data = ready()
        events = [(0, frame(data, 30), OLD), (.6, frame(data, 48), OLD),
                  (.7, frame(data, 51, enabled=False), OLD),
                  (.8, frame(data, 54, session="fresh-arm"), NEW),
                  (.9, frame(data, 57, session="fresh-arm"), NEW)]
        transport, _, _, result = run(events, choose_input("move_to:up", "down"), delay=.3, duration=1.1)
        self.assertTrue(all(p["shoot"] == "none" for _, p, _ in transport.sent))
        self.assertGreaterEqual(result["stale_discarded"], 1)

    def test_weapon_change_stops_old_fire_until_a_reply_observes_new_weapon(self):
        data = ready()
        changed = copy.deepcopy(data)
        changed["player"].update(weapon_type=3, weapon_types=[3])
        events = [(i/30, frame(data if i < 27 else changed, 30+i), OLD) for i in range(48)]
        transport, _, _, result = run(events, choose_input("move_to:up", "right"),
            delay=lambda count: 0 if count == 1 else 3, duration=1.6)
        self.assertTrue(any(p["shoot"] == "right" for when, p, _ in transport.sent if when < .9))
        self.assertTrue(all(p["shoot"] == "none" for when, p, _ in transport.sent if when >= .9))
        self.assertEqual(result["errors"], 0)

    def test_selected_prop_routine_owns_aim_instead_of_parallel_fire_answer(self):
        data = ready()
        data["hazards"] = [fire(x=480, y=280)]
        def choose(payload):
            if "activity" not in payload["questions"]:
                return {"fire": "up"}
            prop = next(c for c in payload["state"]["activity_candidates"] if c["kind"] == "shoot_prop")
            return {"activity": prop["option"], "fire": "up"}
        events = [(i/30, frame(data, 30+i), OLD) for i in range(43)]
        transport, _, _, result = run(events, choose, duration=1.45)
        self.assertTrue(any(p["shoot"] == "right" for _, p, _ in transport.sent))
        self.assertFalse(any(p["shoot"] == "up" for _, p, _ in transport.sent))
        self.assertEqual(result["errors"], 0)

    def test_selected_generic_move_follows_bound_point_and_never_adds_fire(self):
        data = ready()
        nav = PlayerNavigator()
        nav.step(data, 0)
        data["frame"] += 20
        nav.step(data, .7)
        choice = next(c for c in nav.adventure_options if c.key == "move_to:left")
        self.assertTrue(nav.accept_adventure(choice.key, data, .71))
        action = nav.step(data, .72)
        self.assertEqual((action.move, action.shoot), ("left", "none"))
        data["player"].update(x=choice.point[0], y=choice.point[1])
        data["frame"] += 1
        self.assertEqual(nav.step(data, .8).move, "none")

    def test_generic_movement_cannot_collect_pickups_or_enter_doors(self):
        from test_pickups import pickup
        data = ready()
        data["pickups"] = [pickup("free", x=280)]
        parsed = _validated(data)
        self.assertIsNone(room_input_move(data, parsed, (280, 280)))
        for door in data["doors"]:
            self.assertIsNone(room_input_move(data, parsed, (door["x"], door["y"])))
        before = copy.deepcopy(data)
        choices = room_input_candidates(data, parsed)
        self.assertNotIn("move_to:left", {c.key for c in choices})
        self.assertEqual(data, before)


if __name__ == "__main__":
    unittest.main()
