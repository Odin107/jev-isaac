"""Floor-effect inputs, explicit unknown-pill use, and current-position aiming."""
import copy
from pathlib import Path
import sys
import unittest

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).resolve().parents[1]/'src')]
from test_adventure import ready, selected
from test_player_policy import call
from test_player_controller import run, frame, OLD, combat
from test_goals import enemy
from jev_isaac.adventure import candidate_valid
from jev_isaac.player_navigation import PlayerNavigator
from jev_isaac.exploration import _validated, _room_geometry
from jev_isaac.navigation import compute_action
from jev_isaac.state_context import build_game_context


def creep(x=320, y=280):
    return {'kind': 'creep', 'id': 'patch', 'type': 1000, 'variant': 22, 'subtype': 0,
            'surface': 'red', 'x': x, 'y': y, 'vx': 0, 'vy': 0, 'radius': 24, 'timeout': 47}


class CreepPillsAimTests(unittest.TestCase):
    def test_creep_reaches_model_with_explicit_coverage_and_raw_lifetime(self):
        data = combat()
        data['hazards'] = [creep()]
        data['capabilities']['ground_creep'] = 1
        _, payload = call(data, {'goal': 'hold', 'fire': 'none'})
        self.assertEqual(payload['state']['observation']['hazards'][0], creep())
        self.assertIn('exported', payload['state']['game_context']['floor_surfaces']['coverage'])
        data['capabilities'].pop('ground_creep')
        self.assertIn('unavailable', build_game_context(data)['floor_surfaces']['coverage'])

    def test_creep_is_valid_floor_geometry_and_can_be_left_when_underfoot(self):
        data = ready()
        data['player'].update(x=320, y=280, vx=0, vy=0)
        data['hazards'] = [creep()]
        self.assertIsNotNone(_validated(data))
        self.assertTrue(_room_geometry(data, 10)[0])
        action = PlayerNavigator().step(data, 0)
        self.assertNotEqual(action.move, 'none')
        self.assertEqual(action.shoot, 'none')
        self.assertIn('creep', action.status)

    def test_creep_does_not_block_tears_and_combat_dodge_keeps_jev_fire(self):
        data = combat()
        data['hazards'] = [creep()]
        action = compute_action(data, 'hold', fire_direction='right')
        self.assertNotEqual(action.move, 'none')
        self.assertEqual(action.shoot, 'right')
        self.assertIsNotNone(action.override)
        _, payload = call(data, {'goal': 'hold', 'fire': 'right'})
        self.assertEqual(payload['state']['combat_context']['targets'][0]['lanes']['horizontal']['blockers'], [])

    def test_unknown_pill_can_be_chosen_but_never_reveals_a_hidden_effect(self):
        data = ready()
        data['player'].update(pocket_pill=3, pocket_card=0, pill_known=False)
        use = selected(data, 'use_pocket')
        self.assertIsNotNone(use)
        self.assertEqual(use.details['identified'], False)
        self.assertNotIn('effect', use.details)
        self.assertTrue(candidate_valid(data, use))
        data['player'].update(pill_known=True, pill_effect=7)
        self.assertFalse(candidate_valid(data, use))
        known = selected(data, 'use_pocket')
        self.assertNotEqual(known.key, use.key)

    def test_ability_choice_can_use_unknown_pill_once_or_save_it(self):
        for choice, expected in (('none', False), ('ability_0', True)):
            with self.subTest(choice=choice):
                data = combat()
                data['player'].update(pocket_pill=3, pocket_card=0, pill_known=False)
                transport, requests, _, result = run([(i/10, frame(data, 30+i*3), OLD) for i in range(10)],
                    {'goal': 'hold', 'fire': 'none', 'ability': choice})
                self.assertTrue(any('ability' in r['questions'] for r in requests))
                self.assertEqual(any(p.get('interaction') == 'pocket' for _, p, _ in transport.sent), expected)
                self.assertLessEqual(result['interaction_pulses'], 1)

    def test_aim_context_covers_more_than_eight_and_separates_future_positions(self):
        data = combat()
        data['enemies'] = [enemy(f'e{i}', x=100+i*20, y=200) for i in range(20)]
        _, payload = call(data, {'goal': 'hold', 'fire': 'up'})
        self.assertEqual(len(payload['state']['combat_context']['targets']), 20)
        instruction = payload['questions']['combat']['instructions']
        self.assertIn('hypothetical future positions', instruction)
        self.assertIn('current straight-line geometry', instruction)
        for target in payload['state']['combat_context']['targets']:
            self.assertEqual(target['lanes']['vertical']['direction'], 'up')
        self.assertIn('hold firing up', payload['questions']['combat']['criteria']['hold__up'])

    def test_aim_audit_distinguishes_request_position_from_response_position(self):
        data = combat()
        moved = copy.deepcopy(data)
        moved['enemies'][0]['x'] = 160
        events = [(0, frame(data, 30), OLD), (.1, frame(data, 33), OLD)]
        events += [(i/10, frame(moved, 30+i*3), OLD) for i in range(2, 7)]
        _, _, _, result = run(events, {'goal': 'enemy_0', 'fire': 'right'}, delay=.35, duration=.7)
        audit = result['player_decisions'][0]
        self.assertEqual(audit['fire'], 'right')
        self.assertGreater(audit['aim_at_request']['movement_target']['dx'], 0)
        self.assertLess(audit['aim_at_reply']['movement_target']['dx'], 0)
        self.assertEqual(audit['aim_at_request']['aligned_with_chosen_fire'], ['target'])
        self.assertEqual(audit['aim_at_reply']['aligned_with_chosen_fire'], [])


if __name__ == '__main__':
    unittest.main()
