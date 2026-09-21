"""A shallow padding recovery must not cut closer across a rock corner."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_isaac.exploration import _padding_escape
from jev_isaac.navigation import _VECTORS


class EscapeSeparationTests(unittest.TestCase):
    def test_corner_recovery_never_reduces_distance_to_solid_rock(self):
        # A diagonal through the nearby padded face can still move closer to
        # the actual corner. This starts barely clear of the radius-ten player.
        start, velocity = (386.1, 388.0), (0.0, -5.0)
        solid = (340.0, 340.0, 380.0, 380.0)

        def clearance(point):
            x, y = point
            return math.hypot(max(solid[0]-x, 0, x-solid[2]),
                              max(solid[1]-y, 0, y-solid[3]))-10

        for orientation in range(4):
            with self.subTest(orientation=orientation):
                move = _padding_escape(start, velocity, 10, (0, 0, 720, 720),
                                       [(330, 330, 390, 390)], {0: solid})
                self.assertIn(move, _VECTORS)
                self.assertNotEqual(move, "none")
                vector = _VECTORS[move]
                initial = clearance(start)
                self.assertGreater(initial, 0)
                for step in range(121):
                    point = (start[0]+vector[0]*step/10,
                             start[1]+vector[1]*step/10)
                    self.assertGreaterEqual(clearance(point), initial-1e-9)
            start = 360-(start[1]-360), 360+(start[0]-360)
            velocity = -velocity[1], velocity[0]


if __name__ == "__main__":
    unittest.main()
