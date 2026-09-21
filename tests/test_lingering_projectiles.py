"""Clear-room bullet reflexes retain all resource and exit boundaries."""
import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_exploration import door, grid, state
from test_pickups import observed, pickup
from test_clear_room_hazards import npc
from jev_isaac.exploration import ExplorationAction, FloorNavigator
from jev_isaac.navigation import _VECTORS


def bullet(x=360, y=280, vx=-6, vy=0):
    return {"id": "bullet", "type": 9, "x": x, "y": y, "vx": vx, "vy": vy, "radius": 5}


def segment_enters_box(start, vector, box):
    # Independent dense check of the full short direction, including padding.
    return any(box[0] < start[0] + vector[0]*distance/10 < box[2]
               and box[1] < start[1] + vector[1]*distance/10 < box[3]
               for distance in range(241))


class LingeringProjectileTests(unittest.TestCase):
    def test_incoming_bullet_moves_out_of_its_line_without_aiming(self):
        data = state()
        data["projectiles"] = [bullet()]
        before = copy.deepcopy(data)
        action = FloorNavigator().step(data, 0)
        self.assertEqual(action.status, "dodging lingering projectiles")
        self.assertEqual(action.shoot, "none")
        self.assertIsNone(action.stop_reason)
        vector = _VECTORS[action.move]
        # Check the analytic scenario at subframes, not only the chosen label.
        for tenth in range(121):
            time = tenth/10
            player = 320+vector[0]*4*min(time, 6), 280+vector[1]*4*min(time, 6)
            projectile = 360-6*time, 280
            self.assertGreater(math.dist(player, projectile), 15)
        self.assertEqual(data, before)

    def test_departing_bullet_does_not_start_door_or_pickup_pursuit(self):
        for data in (state(doors=[door(2, 85)]), observed(pickup("reward", x=480))):
            data["projectiles"] = [bullet(x=450, vx=6)]
            action = FloorNavigator().step(data, 0)
            self.assertEqual((action.move, action.shoot), ("none", "none"))
            self.assertEqual(action.status, "waiting for lingering hazards")

    def test_dodge_respects_paid_option_and_nonselected_item_boundaries(self):
        for item in (pickup("paid", x=320, y=240, price=15, shop_item=True),
                     pickup("option", x=320, y=240, options_index=1),
                     pickup("item", x=320, y=240, variant=100, subtype=1, collectible_kind=1),
                     pickup("pill", x=320, y=240, variant=70, subtype=3)):
            with self.subTest(item=item):
                data = observed(item)
                data["projectiles"] = [bullet()]
                action = FloorNavigator().step(data, 0)
                self.assertNotEqual(action.move, "none")
                self.assertEqual(action.shoot, "none")
                pad = 10+item["radius"]+8
                box = item["x"]-pad, item["y"]-pad, item["x"]+pad, item["y"]+pad
                self.assertFalse(segment_enters_box((320, 280), _VECTORS[action.move], box))

    def test_dodge_keeps_floor_exits_spikes_rocks_fire_and_npc_boundaries(self):
        blockers = [(grid(320, 240, kind=k, collision=0), (290, 210, 350, 270))
                    for k in (8, 9, 17, 18, 23)]
        blockers.extend(((grid(320, 240), (290, 210, 350, 270)),
                         ({"kind": "fire", "x": 320, "y": 240, "radius": 20}, (282, 202, 358, 278)),
                         ({"kind": "tnt", "x": 320, "y": 240, "radius": 20}, (282, 202, 358, 278))))
        for obstacle, box in blockers:
            data = state()
            data["hazards"] = [obstacle]
            data["projectiles"] = [bullet()]
            action = FloorNavigator().step(data, 0)
            self.assertNotEqual(action.move, "none", obstacle)
            self.assertFalse(segment_enters_box((320, 280), _VECTORS[action.move], box), obstacle)
        data = state()
        data["enemies"] = [npc(x=320, y=238, vy=.1)]
        data["projectiles"] = [bullet()]
        action = FloorNavigator().step(data, 0)
        self.assertNotEqual(action.move, "none")
        self.assertFalse(segment_enters_box((320, 280), _VECTORS[action.move], (282, 200, 358, 277.2)))

    def test_dodge_never_authorizes_a_door_or_moves_outside_room_inset(self):
        data = state(x=570, doors=[door(2, 85)])
        data["hazards"] = [grid(600, 280, kind=16, collision=4)]
        data["projectiles"] = [bullet(x=530, vx=6)]
        action = FloorNavigator().step(data, 0)
        self.assertNotEqual(action.move, "none")
        vector = _VECTORS[action.move]
        self.assertLessEqual(570+24*vector[0], 570)
        self.assertGreaterEqual(280+24*vector[1], 150)
        self.assertLessEqual(280+24*vector[1], 410)

    def test_trapped_or_outside_room_returns_neutral(self):
        for data in (state(), state(x=590)):
            if data["player"]["x"] == 320:
                data["hazards"] = [grid(x, y) for x, y in ((280, 280), (360, 280), (320, 240), (320, 320))]
            data["projectiles"] = [bullet()]
            action = FloorNavigator().step(data, 0)
            self.assertEqual((action.move, action.shoot), ("none", "none"))
            self.assertIsNone(action.stop_reason)

    def test_inactive_incomplete_and_stale_states_never_dodge(self):
        for change in (lambda d: d.update(enabled=False), lambda d: d.update(paused=True),
                       lambda d: d["player"].update(dead=True), lambda d: d.update(truncated=True),
                       lambda d: d["projectiles"][0].update(vx=float("nan"))):
            data = state()
            data["projectiles"] = [bullet()]
            change(data)
            action = FloorNavigator().step(data, 0)
            self.assertEqual((action.move, action.shoot), ("none", "none"))
        data = state()
        data["projectiles"] = [bullet()]
        nav = FloorNavigator()
        self.assertNotEqual(nav.step(data, 0).move, "none")
        self.assertEqual(nav.step(data, .01).move, "none")

    def test_unmodelled_laser_and_bomb_stay_neutral(self):
        for kind in ("laser", "bomb"):
            data = state()
            data["hazards"] = [{"kind": kind, "x": 450, "y": 280, "radius": 10}]
            data["projectiles"] = [bullet()]
            action = FloorNavigator().step(data, 0)
            self.assertEqual((action.move, action.shoot), ("none", "none"))

    def test_active_pickup_plan_is_deferred_and_then_resumed_with_fresh_state(self):
        data = state()
        data["projectiles"] = [bullet()]
        nav = FloorNavigator()
        plan = SimpleNamespace(kind="collect")
        step = Mock(return_value=ExplorationAction(move="right", status="test interaction"))
        nav._adventure = SimpleNamespace(plan=plan, phase="approach", reset_visit=lambda _: None, step=step)
        self.assertEqual(nav.step(data, 0).status, "dodging lingering projectiles")
        step.assert_not_called()
        self.assertIs(nav._adventure.plan, plan)
        data["frame"], data["projectiles"] = 2, []
        self.assertEqual(nav.step(data, .033).status, "test interaction")
        step.assert_called_once()

    def test_already_placed_bomb_retreat_keeps_priority_over_projectile_wait(self):
        data = state()
        data["projectiles"] = [bullet()]
        data["hazards"] = [{"kind": "bomb", "x": 320, "y": 280, "radius": 10}]
        nav = FloorNavigator()
        step = Mock(return_value=ExplorationAction(move="left", status="retreating from placed bomb"))
        nav._adventure = SimpleNamespace(plan=SimpleNamespace(kind="bomb_rock"), phase="retreat",
                                        reset_visit=lambda _: None, step=step)
        action = nav.step(data, 0)
        self.assertEqual((action.move, action.shoot), ("left", "none"))
        self.assertEqual(action.status, "retreating from placed bomb")
        step.assert_called_once()


if __name__ == "__main__":
    unittest.main()
