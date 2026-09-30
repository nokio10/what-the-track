"""Security and round-control checks using real Flask/Socket.IO clients."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.game_app_logic_loader import load_game_app_logic
from tests.generator_service_logic_loader import load_generator_service_logic

ROOT = Path(__file__).resolve().parents[1]
HAS_WEB = all(importlib.util.find_spec(name) for name in ('flask', 'flask_socketio', 'requests'))


class IdentifierBoundaryTests(unittest.TestCase):
    def test_game_ids_reject_trailing_newlines_in_both_services(self):
        for loader in (load_game_app_logic, load_generator_service_logic):
            validate = loader()['is_valid_game_id']
            with self.subTest(loader=loader.__name__):
                self.assertFalse(validate('ABCD\n'))
                self.assertTrue(validate('ABCD-123'))


@unittest.skipUnless(HAS_WEB, 'Install tests/requirements.txt for web integration tests')
class WebSecurityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        previous = os.getcwd()
        try:
            os.chdir(cls.temp.name)
            spec = importlib.util.spec_from_file_location('audit_game_app', ROOT / 'src/game/game_app.py')
            cls.module = importlib.util.module_from_spec(spec)
            with patch.dict(os.environ, {'ADMIN_PASSWORD': 'test-only-password'}):
                spec.loader.exec_module(cls.module)
        finally:
            os.chdir(previous)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.m = self.module
        self.m.game = self.m.GameState()
        self.m.admin_sids.clear()
        self.clients = []

    def tearDown(self):
        for client in self.clients:
            if client.is_connected():
                client.disconnect()

    def client(self, admin=False):
        auth = {'admin': True, 'key': 'test-only-password'} if admin else None
        client = self.m.socketio.test_client(self.m.app, auth=auth)
        self.clients.append(client)
        return client

    def test_untrusted_origin_cannot_open_socket(self):
        response = self.m.app.test_client().get(
            '/socket.io/?EIO=4&transport=polling', headers={'Origin': 'https://untrusted.example'})
        self.assertEqual(response.status_code, 400)

    def test_same_origin_can_open_socket(self):
        response = self.m.app.test_client().get(
            '/socket.io/?EIO=4&transport=polling', headers={'Origin': 'http://localhost'})
        self.assertEqual(response.status_code, 200)

    def test_https_proxy_and_nondefault_port_can_open_socket(self):
        for host, origin, proto in (
            ('quiz.example:5002', 'http://quiz.example:5002', 'http'),
            ('quiz.example', 'https://quiz.example', 'https'),
        ):
            with self.subTest(origin=origin):
                response = self.m.app.test_client().get(
                    '/socket.io/?EIO=4&transport=polling',
                    headers={'Host': host, 'Origin': origin, 'X-Forwarded-Proto': proto})
                self.assertEqual(response.status_code, 200)

    def test_anonymous_client_cannot_create_game(self):
        client = self.client()
        client.emit('admin_create_game')
        self.assertIsNone(self.m.game.game_id)
        self.assertIn('admin_auth_required', [e['name'] for e in client.get_received()])

    def test_wrong_password_cannot_create_game(self):
        client = self.client()
        client.emit('join_admin', {'key': 'wrong'})
        client.emit('admin_create_game')
        self.assertIsNone(self.m.game.game_id)

    def test_authorized_admin_can_create_game(self):
        self.client(admin=True).emit('admin_create_game')
        self.assertIsNotNone(self.m.game.game_id)

    def test_player_does_not_receive_answer_before_reveal(self):
        self.m.game.game_id = 'TEST'
        self.m.game.questions = [{'id': 'q1', 'question': 'A ___', 'answer': 'hidden-answer'}]
        self.m.game.current_q_index = 0
        self.m.game.current_phase = 'question'
        self.m.game.is_active = True
        player = self.client()
        player.emit('join_game', {'name': 'Alice', 'player_id': 'alice-private-id'})
        events = player.get_received()
        self.assertIn('new_question', [e['name'] for e in events])
        self.assertNotIn('hidden-answer', str(events))
        self.assertNotIn('admin_update', [e['name'] for e in events])

    def test_other_browser_cannot_take_existing_player_name(self):
        self.m.game.game_id = 'TEST'
        owner = self.client()
        owner.emit('join_game', {'name': 'Alice', 'player_id': 'owner-id'})
        owner.disconnect()
        other = self.client()
        other.emit('join_game', {'name': 'Alice', 'player_id': 'other-id'})
        self.assertIn('join_error', [e['name'] for e in other.get_received()])
        self.assertEqual(len(self.m.game.players), 1)

    def test_round_scoring_reconnect_and_game_completion(self):
        import json
        admin = self.client(admin=True)
        admin.emit('admin_create_game')
        gid = self.m.game.game_id
        questions = [{'id': 'q1', 'question': 'White ___', 'answer': 'snow'}]
        Path(self.m.MEDIA_ROOT, f'{gid}-questions.json').write_text(json.dumps(questions))
        admin.emit('admin_start_round')
        alice, bob, waiting = self.client(), self.client(), self.client()
        for client, name in ((alice, 'Alice'), (bob, 'Bob'), (waiting, 'Charlie')):
            client.emit('join_game', {'name': name, 'player_id': name + '-private'})
        admin.emit('admin_next_question', {'from_index': -1})
        self.assertEqual(self.m.game.current_phase, 'question')
        alice.emit('submit_answer', {'answer': 'snow'})
        self.assertTrue(all(p['last_answer'] is None for p in self.m.game.players.values()))
        admin.emit('admin_audio_finished')
        alice.emit('submit_answer', {'answer': 'snow'})
        bob.emit('submit_answer', {'answer': 'rain'})
        alice.disconnect()
        alice = self.client()
        alice.emit('join_game', {'name': 'Alice', 'player_id': 'Alice-private'})
        restored = next(p for p in self.m.game.players.values() if p['name'] == 'Alice')
        self.assertEqual(restored['last_answer'], 'snow')
        admin.emit('admin_show_answer')
        scores = {p['name']: p['score'] for p in self.m.game.players.values()}
        self.assertEqual(scores, {'Alice': 2, 'Bob': 0, 'Charlie': 0})
        admin.emit('admin_show_answer')
        self.assertEqual(restored['score'], 2, 'Repeated reveal must not award points twice')
        admin.emit('admin_audio_finished')
        admin.emit('admin_next_question', {'from_index': 0})
        self.assertEqual(self.m.game.current_phase, 'finished')
        self.assertEqual(self.m.game.final_results['winners'], [{'name': 'Alice', 'score': 2}])

    def test_vip_cannot_skip_answer_audio_before_admin_finishes_it(self):
        self.m.game.game_id = 'TEST'
        player = self.client()
        player.emit('join_game', {'name': 'Alice', 'player_id': 'owner-id'})
        self.m.game.questions = [
            {'id': 'q1', 'question': 'A ___', 'answer': 'first'},
            {'id': 'q2', 'question': 'B ___', 'answer': 'second'},
        ]
        self.m.game.current_q_index = 0
        self.m.game.current_phase = 'answer'
        self.m.game.is_active = True
        self.m.game.next_allowed = False
        player.emit('player_next_question')
        self.assertEqual(self.m.game.current_q_index, 0)
        self.client(admin=True).emit('admin_audio_finished')
        player.emit('player_next_question')
        self.assertEqual(self.m.game.current_q_index, 1)