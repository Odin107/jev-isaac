import unittest

from test_navigation import state, enemy
from jev_isaac.navigation import compute_action


class ExplosiveLaneTests(unittest.TestCase):
    def test_opportunistic_shots_do_not_cross_observed_bomb_or_movable_tnt(self):
        for kind in ("tnt", "bomb"):
            for goal in ("engage", "evade", "hold"):
                with self.subTest(kind=kind, goal=goal):
                    data = state(200, 250)
                    data["enemies"] = [enemy(400, 250)]
                    data["hazards"] = [{"kind": kind, "x": 300, "y": 250,
                                        "vx": 0, "vy": 0, "radius": 15}]
                    action = compute_action(data, goal, "target" if goal == "engage" else None)
                    self.assertEqual(action.shoot, "none")


if __name__ == "__main__":
    unittest.main()
