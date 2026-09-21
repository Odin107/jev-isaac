"""Navigation status is carried only by a neutral control release."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from jev_isaac.cli import save_navigation_checkpoint
from jev_isaac.demo import sample
from jev_isaac.protocol import Observation, encode_action


class NavigationReleaseTests(unittest.TestCase):
    def test_navigation_reason_encodes_only_with_explicit_neutral_release(self):
        observation = Observation(sample(30))
        packet = json.loads(encode_action(observation, 'none', 'none', 1,
                            floor_mode=False, stop_reason='navigation'))
        self.assertEqual(packet['stop_reason'], 'navigation')
        self.assertFalse(packet['floor_mode'])
        self.assertEqual(packet['frame'], 30)
        self.assertNotIn('stop_reason', json.loads(encode_action(observation, 'left', 'up')))

    def test_reason_cannot_accompany_active_controls_or_unknown_status(self):
        base = dict(move='none', shoot='none', hold_frames=1, floor_mode=False,
                    stop_reason='navigation')
        for change in ({'move': 'left'}, {'shoot': 'up'}, {'hold_frames': 2},
                       {'floor_mode': True}, {'floor_mode': None},
                       {'stop_reason': True}, {'stop_reason': 'unknown'},
                       {'interaction': 'bomb', 'interaction_id': 'pulse'},
                       {'move_frames': 1, 'move_distance': 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                encode_action(Observation(sample(30)), **(base | change))

    def test_checkpoint_is_immediately_readable_without_overwriting_final_report(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / 'nested/live.json'
            report.parent.mkdir()
            report.write_text('previous final report', encoding='utf-8')
            snapshot = {'reason': 'no safe route to open door', 'observation': sample(30)}
            save_navigation_checkpoint(report, snapshot)
            destination = report.with_name('live-navigation.json')
            recorded = json.loads(destination.read_text(encoding='utf-8'))
            self.assertEqual(recorded['navigation_failure'], snapshot)
            self.assertIn('recorded_at', recorded)
            self.assertEqual(report.read_text(encoding='utf-8'), 'previous final report')
            snapshot['reason'] = 'door traversal timed out'
            save_navigation_checkpoint(report, snapshot)
            self.assertEqual(json.loads(destination.read_text(encoding='utf-8'))['navigation_failure'], snapshot)
            self.assertFalse(destination.with_suffix('.json.tmp').exists())
        save_navigation_checkpoint(None, {})
