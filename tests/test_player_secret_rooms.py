"""Observed secret entrances are choices, never guessed hidden map facts."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).resolve().parents[1]/'src')]
from test_adventure import ready
from test_exploration import door
from test_player_navigation import offers
from test_player_policy import call
from test_player_controller import run, frame, OLD
from jev_isaac.player_navigation import PlayerNavigator
from jev_isaac.exploration import _validated, FloorNavigator


def recorded():
    data = json.loads((Path(__file__).parent/'fixtures/player-secret-room-stop.json').read_text(encoding='utf-8-sig'))
    data.update(enabled=True, paused=False)
    return data


class SecretRoomTests(unittest.TestCase):
    def test_open_secret_and_supersecret_entrances_reach_model_and_execute(self):
        for kind, label in ((7, 'secret'), (8, 'supersecret')):
            with self.subTest(kind=kind):
                data = ready()
                data['doors'] = [door(0, 83, kind=kind), door(2, 85)]
                nav = PlayerNavigator()
                offered = [c.as_dict() for c in offers(nav, data)]
                selected = next(c for c in offered if c['key'] == 'enter:0:83')
                self.assertIn(label, selected['description'])
                self.assertEqual(selected['cost'], {})
                data['_adventure_options'] = offered
                chosen, payload = call(data, {'activity': f'action_{offered.index(selected)}'})
                self.assertTrue(nav.accept_adventure(chosen.target_id, data, .7))
                data['frame'] += 1
                action = nav.step(data, .8)
                self.assertEqual((action.move, action.shoot), ('left', 'none'))
                self.assertTrue(any(c['key'] == selected['key'] for c in payload['state']['activity_candidates']))

    def test_closed_locked_and_unobserved_secrets_are_not_invented_actions(self):
        for secret in ([], [door(0, 83, kind=7, opened=False)],
                       [dict(door(0, 83, kind=8), locked=True)]):
            data = ready()
            data['doors'] = secret + [door(2, 85)]
            choices = offers(PlayerNavigator(), data)
            self.assertFalse(any(c.target_id == '83' for c in choices))
            self.assertFalse(any(c.kind in ('bomb_secret', 'search_secret', 'unlock_door') for c in choices))

    def test_exact_secret_room_stop_now_offers_exit_and_survives_rearm(self):
        data, nav = recorded(), PlayerNavigator()
        choices = offers(nav, data)
        self.assertIsNone(nav.stop_reason)
        exit_choice = next(c for c in choices if c.key == 'enter:2:70')
        self.assertTrue(nav.accept_adventure(exit_choice.key, data, .7))
        data['frame'] += 1
        self.assertEqual(nav.step(data, .8).move, 'right')
        data['session'] += ':rearm'
        data['frame'] += 1
        fresh = nav.rearmed(data)
        self.assertTrue(fresh.allow_secret)
        self.assertIsNone(fresh.stop_reason)
        self.assertIsNone(fresh._intent)
        self.assertEqual(fresh.stats['rooms_visited'], nav.stats['rooms_visited'])

    def test_actual_controller_requests_activity_in_recorded_secret_room(self):
        data = recorded()
        def choose(payload):
            if 'activity' not in payload['questions']:
                return {'fire': 'none'}
            choice = next(c for c in payload['state']['activity_candidates'] if c['key'] == 'enter:2:70')
            return {'activity': choice['option']}
        transport, requests, _, result = run([(i/10, frame(data, data['frame']+i*3), OLD) for i in range(11)], choose)
        self.assertTrue(requests)
        self.assertEqual(result['errors'], 0)
        self.assertNotEqual(result['stop_reason'], 'unsupported room type')
        self.assertTrue(any(p['move'] == 'right' for _, p, _ in transport.sent))

    def test_roundtrip_remembers_origin_exits_and_does_not_force_return(self):
        data, nav = ready(), PlayerNavigator()
        origin_index = data['floor']['room_index']
        data['doors'] = [door(0, 83, kind=8), door(2, 85)]
        offers(nav, data)
        self.assertTrue(nav.accept_adventure('enter:0:83', data, .7))
        incoming = copy.deepcopy(data)
        incoming.update(frame=data['frame']+1, room_id='secret-arrival')
        incoming['floor'].update(room_index=83, room_list_index=33)
        incoming['room']['type'] = 8
        incoming['doors'] = [door(2, origin_index)]
        nav.step(incoming, .8)
        self.assertIsNone(nav.stop_reason)
        finished = nav.player_events[-1]
        self.assertEqual(finished['room_index'], origin_index)
        incoming['frame'] += 18
        nav.step(incoming, 1.4)
        back = next(c for c in nav.adventure_options if c.key == f'enter:2:{origin_index}')
        self.assertTrue(back.details['visited'])
        self.assertTrue(back.details['destination_doors_inspected'])
        self.assertTrue(any(d['target_index'] == 85 and not d['visited']
                            for d in back.details['remembered_destination_exits']))
        self.assertIsNone(nav._pending)

    def test_failed_route_is_exposed_and_snapshot_is_copied_for_diagnosis(self):
        data, nav = ready(), PlayerNavigator()
        offers(nav, data)
        key = next(c.key for c in nav.adventure_options if c.kind == 'enter_door')
        self.assertTrue(nav.accept_adventure(key, data, .7))
        original = copy.deepcopy(data)
        nav._done(data, .8, 'recorded route failure', failed=True)
        data['frame'] += 15
        nav.step(data, 1.3)
        retry = next(c for c in nav.adventure_options if c.key == key)
        self.assertEqual(retry.details['recent_times_selected_from_here'], 1)
        self.assertEqual(retry.details['last_outcome_from_here'], 'recorded route failure')
        self.assertEqual(nav.activity_failures[0]['observation'], original)
        self.assertNotIn('activity_failures', nav.decision_context())

    def test_legacy_mode_and_other_special_rooms_keep_their_existing_scope(self):
        data = ready()
        data['doors'] = [door(0, 83, kind=8)]
        self.assertEqual(_validated(data)[-1], ())
        secret = recorded()
        old = FloorNavigator(adventure_mode=True)
        old.observe(secret)
        self.assertEqual(old.stop_reason, 'unsupported room type')
        secret['room']['type'] = 29
        new = PlayerNavigator()
        new.observe(secret)
        self.assertEqual(new.stop_reason, 'unsupported room type')

    def test_activity_failure_checkpoints_geometry_without_stopping_controller(self):
        data, checkpoints = ready(), []
        def choose(payload):
            return {'activity': next(c['option'] for c in payload['state']['activity_candidates']
                                     if c['kind'] == 'enter_door')}
        with patch('jev_isaac.player_navigation._door_move', return_value=None):
            _, _, _, result = run([(i/10, frame(data, 30+i*3), OLD) for i in range(14)],
                                   choose, checkpoint=checkpoints.append)
        self.assertTrue(checkpoints)
        self.assertEqual(checkpoints[0]['kind'], 'activity_failure')
        self.assertEqual(checkpoints[0]['observation']['hazards'], data['hazards'])
        self.assertEqual(result['navigation_stops'], 0)
        self.assertEqual(result['errors'], 0)
        self.assertEqual(result['stop_reason'], 'duration reached')


if __name__ == '__main__':
    unittest.main()
