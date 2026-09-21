"""Recorded post-explosion doorway drift; no game or provider calls."""
import copy
from collections import deque
import json
import math
from pathlib import Path
import sys
import unittest

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).resolve().parents[1]/'src')]
from jev_isaac.player_navigation import PlayerNavigator
from jev_isaac.switches import SwitchNavigator
from jev_isaac.navigation import _VECTORS
from test_floor_drift_regression import physics_substep
from test_pickups import pickup


def recorded():
    data = json.loads((Path(__file__).parent/'fixtures/player-post-tnt-switch.json').read_text(encoding='utf-8-sig'))
    data.update(enabled=True, paused=False)
    return data


class PlayerSwitchBoundaryTests(unittest.TestCase):
    def test_exact_recorded_doorway_drift_recovers_inward_without_shooting(self):
        data = recorded()
        before = copy.deepcopy(data)
        self.assertAlmostEqual(data['player']['x']-570, 2.68920898438)
        nav = SwitchNavigator()
        action = nav.step(data, 0, selected_switch=22, allow_demolition=False)
        self.assertIsNone(action.stop_reason)
        self.assertEqual((action.move, action.shoot), ('left', 'none'))
        self.assertEqual(nav.target[0], 22)
        self.assertEqual(data, before)

    def test_unselected_door_is_not_crossed_and_wait_does_not_start_recovery(self):
        data, nav = recorded(), PlayerNavigator()
        nav.step(data, 0)
        data['frame'] += 18
        action = nav.step(data, .6)
        self.assertEqual((action.move, action.shoot), ('none', 'none'))
        self.assertIsNone(nav._pending)
        self.assertTrue(nav.accept_adventure('switch:22', data, .7))
        data['frame'] += 1
        action = nav.step(data, .8)
        self.assertEqual((action.move, action.shoot), ('left', 'none'))
        self.assertIsNone(nav._pending)

    def test_dangerous_unknown_closed_or_locked_door_padding_remains_blocked(self):
        changes = [dict(open=False), dict(locked=True), dict(curse_room_door=True),
                   dict(curse_room_door=None), dict(target_type=8), dict(target_type=10)]
        for change in changes:
            with self.subTest(change=change):
                data = recorded()
                next(d for d in data['doors'] if d['slot'] == 2).update(change)
                action = SwitchNavigator().step(data, 0, selected_switch=22, allow_demolition=False)
                self.assertIsNotNone(action.stop_reason)
                self.assertEqual((action.move, action.shoot), ('none', 'none'))

    def test_deep_fast_or_paid_overlap_cannot_use_doorway_recovery(self):
        changes = [lambda d: d['player'].update(x=575),
                   lambda d: d['player'].update(x=581),
                   lambda d: d['player'].update(vx=3),
                   lambda d: d['pickups'].append(pickup('paid-overlap', x=573, y=293, price=5, shop_item=True))]
        for change in changes:
            with self.subTest(change=change):
                data = recorded()
                change(data)
                action = SwitchNavigator().step(data, 0, selected_switch=22, allow_demolition=False)
                self.assertIsNotNone(action.stop_reason)
                self.assertEqual((action.move, action.shoot), ('none', 'none'))

    def test_recovery_remains_bounded_if_the_game_does_not_move(self):
        data, nav = recorded(), SwitchNavigator()
        self.assertIsNone(nav.step(data, 0, selected_switch=22, allow_demolition=False).stop_reason)
        data['frame'] += 121
        self.assertEqual(nav.step(data, 4.01, selected_switch=22, allow_demolition=False).stop_reason,
                         'room switch approach stalled')

    def test_recorded_room_reaches_selected_plate_with_delayed_momentum(self):
        # Stress replay, not an Isaac physics emulator. Only plate activation
        # is supplied once the player actually approaches its observed center.
        for delay in (1, 2):
            with self.subTest(delay=delay):
                data, nav = recorded(), PlayerNavigator()
                nav.step(data, 0)
                data['frame'] += 18
                nav.step(data, .6)
                self.assertTrue(nav.accept_adventure('switch:22', data, .7))
                pending = deque(['none']*delay)
                for tick in range(1, 450):
                    data['frame'] += 1
                    action = nav.step(data, .7+tick/30)
                    self.assertFalse(any(e['event'] == 'failed' for e in nav.player_events),
                                     (nav.player_events, {k: data['player'][k] for k in ('x', 'y', 'vx', 'vy')}))
                    self.assertEqual(action.shoot, 'none')
                    self.assertIsNone(nav._pending)
                    pending.append(action.move)
                    applied = _VECTORS[pending.popleft()]
                    for _ in range(2):
                        physics_substep(data['player'], applied)
                        self.assertLessEqual(data['player']['x'], 572.689209)
                        self.assertGreaterEqual(data['player']['x'], 70)
                        self.assertGreaterEqual(data['player']['y'], 150)
                    if math.dist((data['player']['x'], data['player']['y']), (320, 160)) <= 8:
                        data['switches'][0]['state'] = 3
                        data['frame'] += 1
                        action = nav.step(data, .7+(tick+1)/30)
                        self.assertEqual(nav.player_events[-1]['event'], 'finished')
                        self.assertEqual((action.move, action.shoot), ('none', 'none'))
                        break
                else:
                    self.fail('failed to reach the selected plate')


if __name__ == '__main__':
    unittest.main()
