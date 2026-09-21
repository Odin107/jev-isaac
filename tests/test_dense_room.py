"""Large-room geometry and incomplete-observation recovery without API calls."""
import copy
import unittest

from test_player_controller import combat, frame, run, OLD, NEW
from jev_isaac.combat import build_combat_context
from jev_isaac.combat_stall import no_living_enemies
from jev_isaac.exploration import _validated
from jev_isaac.navigation import compute_action
from jev_isaac.protocol import MAX_HAZARDS, Observation, encode_action


class DenseRoomTests(unittest.TestCase):
    def test_geometry_after_old_cap_is_used_by_every_local_consumer(self):
        data = combat()
        data['player'].update(x=320, y=280)
        data['hazards'] = [dict(kind='grid', type=7, collision=0, radius=20,
                                index=i, x=80+i%13*40, y=160+i//13*40)
                           for i in range(200)]
        data['hazards'].append(dict(kind='grid', type=2, collision=3, radius=20,
                                   index=300, x=400, y=280))
        self.assertIsNotNone(_validated(data, allow_shop=True, allow_secret=True))
        context = build_combat_context(data)
        self.assertIn(300, context['targets'][0]['lanes']['horizontal']['blockers'])
        self.assertEqual(compute_action(data, 'hold', fire_direction='up').shoot, 'up')
        data['enemies'] = []
        self.assertTrue(no_living_enemies(data))
        data['hazards'] *= 3
        self.assertGreater(len(data['hazards']), MAX_HAZARDS)
        self.assertIsNone(_validated(data))
        self.assertFalse(no_living_enemies(data))

    def test_incomplete_geometry_waits_through_pause_and_recovers_only_on_fresh_f8(self):
        data = combat()
        incomplete = copy.deepcopy(data)
        incomplete['capabilities']['observation_recovery'] = 1
        incomplete.update(truncated=True, truncated_arrays={'hazards': True})
        events = [(0, frame(data, 30), OLD), (.1, frame(data, 33), OLD),
                  (.2, frame(incomplete, 36), OLD),
                  (.3, frame(incomplete, 36, paused=True, enabled=False), OLD),
                  (1.3, frame(incomplete, 36, paused=True, enabled=False), OLD),
                  (1.5, frame(data, 37), OLD),  # complete data alone cannot restart control
                  (1.7, frame(data, 38, session='fresh-arm'), NEW),
                  (1.8, frame(data, 41, session='fresh-arm'), NEW),
                  (1.9, frame(data, 44, session='fresh-arm'), NEW)]
        transport, requests, logs, result = run(events, {'goal': 'hold', 'fire': 'right'}, duration=2.1)
        self.assertEqual(result['stop_reason'], 'duration reached')
        self.assertEqual((result['navigation_stops'], result['navigation_rearms']), (1, 1))
        self.assertEqual(result['errors'], 0)
        releases = [p for _, p, _ in transport.sent if p.get('stop_reason') == 'observation']
        self.assertEqual(len(releases), 1)
        self.assertFalse(any(.21 <= t < 1.7 for t, _, _ in transport.sent))
        self.assertTrue(any(t >= 1.7 and p['shoot'] == 'right' for t, p, _ in transport.sent))
        self.assertTrue(all(not r['state']['observation'].get('truncated') for r in requests))
        self.assertAlmostEqual(transport.now, 2.1, delta=.011)

    def test_rearming_into_still_incomplete_room_stays_open_without_paid_requests(self):
        data = combat()
        data.update(truncated=True, truncated_arrays={'hazards': True})
        events = [(0, frame(data, 30), OLD),
                  (.1, frame(data, 31, enabled=False), OLD),
                  (.2, frame(data, 32, session='fresh-arm'), NEW)]
        _, requests, _, result = run(events, {'goal': 'hold'}, duration=.5)
        self.assertEqual(result['stop_reason'], 'duration reached')
        self.assertEqual(result['navigation_stops'], 2)
        self.assertEqual(requests, [])

    def test_observation_release_cannot_authorize_any_input(self):
        data = Observation(combat())
        encode_action(data, 'none', 'none', 1, floor_mode=False, stop_reason='observation')
        for changes in ({'move': 'left'}, {'shoot': 'up'}, {'floor_mode': True}, {'hold_frames': 2},
                        {'interaction': 'active', 'interaction_id': 'pulse'}):
            arguments = dict(move='none', shoot='none', hold_frames=1, floor_mode=False,
                             stop_reason='observation')
            arguments.update(changes)
            with self.assertRaises(ValueError):
                encode_action(data, **arguments)


if __name__ == '__main__':
    unittest.main()
