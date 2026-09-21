"""Delayed rewards must become real choices before an old departure executes."""
import copy
import unittest

from test_adventure import ready
from test_exploration import grid
from test_pickups import pickup
from test_player_navigation import offers
from test_player_controller import run, frame, OLD
from jev_isaac.adventure import AdventureCandidate
from jev_isaac.player_navigation import PlayerNavigator


def item(ident="reward", **changes):
    return pickup(ident, variant=100, subtype=1, collectible_kind=1, **changes)


class RewardTimingTests(unittest.TestCase):
    def test_boss_clear_waits_for_late_reward_but_jev_can_still_leave_it(self):
        data = ready(clear=False, kind=5)
        data["hazards"] = [grid(320, 360, kind=17, collision=0)]
        data["enemies"] = [{"id": "boss", "x": 480, "y": 280, "vx": 0, "vy": 0,
                            "hp": 10, "radius": 12, "vulnerable": True}]
        nav = PlayerNavigator(continue_floors=True)
        nav.step(data, 0)
        data["frame"] += 60
        data["room"]["clear"], data["enemies"] = True, []
        nav.step(data, 2)
        self.assertEqual(nav.adventure_options, ())
        data["frame"] += 6
        data["pickups"] = [item()]
        nav.step(data, 2.2)
        self.assertEqual(nav.adventure_options, ())
        data["frame"] += 12
        nav.step(data, 2.6)
        self.assertTrue(any(c.kind == "collect" for c in nav.adventure_options))
        descent = next(c for c in nav.adventure_options if c.kind == "descend")
        self.assertTrue(nav.accept_adventure(descent.key, data, 2.7))
        self.assertEqual(nav._intent.kind, "descend")

    def test_new_reward_invalidates_cached_choices_before_reply_and_versions_rebuilt_door(self):
        data, nav = ready(), PlayerNavigator()
        old = next(c for c in offers(nav, data) if c.kind == "enter_door")
        data["frame"] += 1
        data["pickups"] = [item()]
        nav.observe(data)
        revision = nav._room_choice_revision
        self.assertEqual(nav.adventure_options, ())
        self.assertFalse(nav.accept_adventure(old.key, data, .7))
        nav.observe(data)  # Controller.observe + step.observe must not reset time.
        self.assertEqual(nav._room_choice_revision, revision)
        data["frame"] += 12
        nav.step(data, 1.1)
        new = next(c for c in nav.adventure_options if c.key == old.key)
        self.assertNotEqual(new.as_dict(), old.as_dict())
        self.assertGreater(new.details["room_choice_revision"], old.details["room_choice_revision"])
        self.assertTrue(any(c.kind == "collect" for c in nav.adventure_options))

    def test_positive_wait_countdown_and_bounce_do_not_extend_readiness_timer(self):
        data, nav = ready(item(wait=120)), PlayerNavigator()
        offers(nav, data)
        started, revision = nav._reward_wait_started_frame, nav._room_choice_revision
        for _ in range(3):
            data["frame"] += 20
            data["pickups"][0].update(wait=data["pickups"][0]["wait"]-20,
                                      x=data["pickups"][0]["x"]+1, vx=1)
            nav.step(data, data["frame"]/30)
            self.assertEqual(nav.adventure_options, ())
        self.assertEqual(nav._reward_wait_started_frame, started)
        self.assertEqual(nav._room_choice_revision, revision)
        data["frame"] = started+90
        nav.step(data, data["frame"]/30)
        self.assertTrue(nav.adventure_options)
        self.assertTrue(nav.decision_context()["reward_observation"]["readiness_wait_expired"])
        self.assertFalse(any(c.kind == "collect" for c in nav.adventure_options))

    def test_readiness_changes_refresh_choices_after_quiet_period(self):
        for wait, can_pickup in ((10, True), (0, False)):
            with self.subTest(wait=wait, can_pickup=can_pickup):
                data, nav = ready(item(wait=wait)), PlayerNavigator()
                data["player"]["can_pickup_items"] = can_pickup
                offers(nav, data)
                self.assertEqual(nav.adventure_options, ())
                data["frame"] += 1
                data["pickups"][0]["wait"] = 0
                data["player"]["can_pickup_items"] = True
                nav.step(data, .7)
                self.assertEqual(nav.adventure_options, ())
                data["frame"] += 12
                nav.step(data, 1.1)
                self.assertTrue(any(c.kind == "collect" for c in nav.adventure_options))

    def test_new_reward_revokes_uncommitted_door_or_descent_without_collecting(self):
        for kind in ("enter_door", "descend"):
            with self.subTest(kind=kind):
                data, nav = ready(kind=5), PlayerNavigator(continue_floors=True)
                data["hazards"] = [grid(320, 360, kind=17, collision=0)]
                selected = next(c for c in offers(nav, data) if c.kind == kind)
                self.assertTrue(nav.accept_adventure(selected.key, data, .7))
                # A previous curse-room crossing can leave this historical
                # executor phase behind; it cannot commit an ordinary door.
                if kind == "enter_door":
                    nav._adventure.phase = "crossing"
                data["frame"] += 1
                data["pickups"] = [item()]
                action = nav.step(data, .8)
                self.assertEqual((action.move, action.shoot), ("none", "none"))
                self.assertIsNone(nav._intent)
                self.assertIsNone(nav._pending)
                self.assertEqual(nav.player_events[-1]["event"], "canceled")
                self.assertFalse(nav.descent_requested)

    def test_new_reward_keeps_committed_blast_curse_crossing_and_descent(self):
        for kind, phase, descended in (("bomb_rock", "retreat", False),
                                       ("leave_curse", "crossing", False),
                                       ("descend", "approach", True)):
            with self.subTest(kind=kind):
                data, nav = ready(), PlayerNavigator()
                offers(nav, data)
                selected = AdventureCandidate("committed", kind, "target", (320, 280), {}, "chosen")
                nav._intent = nav._adventure.plan = selected
                nav._adventure.phase, nav._adventure.descent_requested = phase, descended
                data["frame"] += 1
                data["pickups"] = [item()]
                nav.observe(data)
                self.assertIs(nav._intent, selected)
                self.assertIs(nav._adventure.plan, selected)

    def test_collection_followup_drop_extends_quiet_period_and_is_offered(self):
        data, nav = ready(item("first")), PlayerNavigator()
        selected = next(c for c in offers(nav, data) if c.kind == "collect")
        nav.accept_adventure(selected.key, data, .7)
        data["frame"] += 1
        data["pickups"] = []
        nav.step(data, .8)
        self.assertIsNone(nav._intent)
        data["frame"] += 10
        data["pickups"] = [item("followup")]
        nav.step(data, 1.2)
        data["frame"] += 3
        nav.step(data, 1.3)
        self.assertEqual(nav.adventure_options, ())
        data["frame"] += 9
        nav.step(data, 1.6)
        self.assertTrue(any(c.target_id == "followup" for c in nav.adventure_options))

    def test_observed_room_transition_is_not_canceled_as_a_reward_change(self):
        data, nav = ready(), PlayerNavigator()
        selected = next(c for c in offers(nav, data) if c.kind == "enter_door")
        nav.accept_adventure(selected.key, data, .7)
        incoming = copy.deepcopy(data)
        incoming.update(room_id="new-room", frame=data["frame"]+1, pickups=[item()])
        incoming["floor"]["room_index"] = selected.details["target_index"]
        nav.observe(incoming)
        self.assertIsNone(nav.stop_reason)
        self.assertEqual(nav.player_events[-1]["reason"], "observed room transition")
        self.assertEqual(nav.player_events[-1]["event"], "finished")

    def test_missing_optional_pickup_data_does_not_erase_observed_reward_snapshot(self):
        data, nav = ready(item()), PlayerNavigator()
        offers(nav, data)
        snapshot, revision = nav._reward_signature, nav._room_choice_revision
        data["frame"] += 1
        data.pop("pickups")
        data["capabilities"].pop("pickup_collection")
        nav.observe(data)
        self.assertIsNone(nav.stop_reason)
        self.assertEqual(nav._reward_signature, snapshot)
        self.assertEqual(nav._room_choice_revision, revision)

    def test_first_available_pickup_data_refreshes_choices_made_with_unknown_contents(self):
        data, nav = ready(), PlayerNavigator()
        data.pop("pickups")
        data["capabilities"].pop("pickup_collection")
        offers(nav, data)
        self.assertTrue(nav.adventure_options)
        data["frame"] += 1
        data["pickups"] = [item()]
        data["capabilities"]["pickup_collection"] = 1
        nav.observe(data)
        self.assertIsNone(nav.stop_reason)
        self.assertEqual(nav.adventure_options, ())
        data["frame"] += 12
        nav.step(data, 1.1)
        self.assertTrue(any(c.kind == "collect" for c in nav.adventure_options))


class RewardTimingControllerTests(unittest.TestCase):
    def test_delayed_reward_discards_pending_door_reply_and_reaches_next_model_request(self):
        empty, reward = ready(), ready(item())
        events = [(index/30, frame(empty if index < 21 else reward, 1+index), OLD)
                  for index in range(48)]
        def choose(payload):
            candidate = next((c for c in payload["state"].get("activity_candidates", [])
                              if c["kind"] == "enter_door"), None)
            return {"activity": candidate["option"] if candidate and not
                    payload["state"]["observation"]["pickups"] else "wait", "fire": "none"}
        transport, requests, _, result = run(events, choose, delay=.2, duration=1.6)
        self.assertEqual(result["errors"], 0)
        self.assertFalse(result["player_decisions"][0]["accepted"])
        self.assertGreaterEqual(result["stale_discarded"], 1)
        self.assertTrue(any(c["kind"] == "collect" for request in requests[1:]
                            for c in request["state"].get("activity_candidates", [])))
        self.assertTrue(all(packet["move"] == "none" for _, packet, _ in transport.sent))


if __name__ == "__main__":
    unittest.main()
