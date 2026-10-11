"""Voice routing and state changes, without network or audio hardware."""
import threading
import unittest
from unittest.mock import Mock, patch

import config
import voice


class VoiceTests(unittest.TestCase):
    def setUp(self):
        voice._last_state = None
        voice._last_speak_time = 0.
        voice._pending_text = None

    def test_drone_speech_never_needs_cloud_or_credentials(self):
        with patch.object(config, 'VIDEO_SOURCE', 'drone'), \
             patch('voice.shutil.which', return_value='/usr/bin/say'), \
             patch('voice.subprocess.run') as run, patch('voice._init') as cloud:
            voice._speak('Target acquired.')
        cloud.assert_not_called()
        self.assertEqual(run.call_args.args[0], ['/usr/bin/say', 'Target acquired.'])
        self.assertEqual(run.call_args.kwargs['timeout'], 10)

    def test_cloud_failure_falls_back_to_offline_speech(self):
        client = Mock()
        client.text_to_speech.convert.side_effect = ConnectionError('offline')
        with patch.object(config, 'VIDEO_SOURCE', 'webcam'), \
             patch('voice._init', return_value=True), patch.object(voice, '_eleven', client), \
             patch('voice._speak_local', return_value=True) as local:
            voice._speak('Target lost.')
        local.assert_called_once_with('Target lost.')
        self.assertEqual(client.text_to_speech.convert.call_args.kwargs['request_options']['max_retries'], 0)

    def test_missing_cloud_credentials_uses_offline_speech(self):
        with patch.object(config, 'VIDEO_SOURCE', 'webcam'), \
             patch('voice._init', return_value=False), patch('voice._speak_local') as local:
            voice._speak('Target acquired.')
        local.assert_called_once_with('Target acquired.')

    def test_speech_is_nonblocking_and_does_not_overlap(self):
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        def speak(text):
            entered.set()
            release.wait(2)
            finished.set()
        lock = threading.Lock()
        with patch.object(voice, '_speech_lock', lock), patch('voice._speak', side_effect=speak):
            try:
                self.assertTrue(voice._speak_async('First'))
                self.assertTrue(entered.wait(1))
                self.assertFalse(voice._speak_async('Second'))
            finally:
                release.set()
                self.assertTrue(finished.wait(1))
                self.assertTrue(lock.acquire(timeout=1))
                lock.release()

    def test_callout_waits_through_cooldown_then_speaks_stable_state(self):
        with patch('voice._speak_async', return_value=True) as speak, patch('voice._init') as init:
            with patch('voice.time.monotonic', return_value=10.):
                voice.announce('LOCKED')
            speak.reset_mock()
            with patch('voice.time.monotonic', return_value=11.):
                voice.announce('REACHED')
            speak.assert_not_called()
            with patch('voice.time.monotonic', return_value=13.1):
                voice.announce('REACHED')
                voice.announce('REACHED')
            speak.assert_called_once()
            init.assert_not_called()  # Client setup belongs on the worker.

    def test_busy_speaker_does_not_replay_an_obsolete_acquisition(self):
        with patch('voice._speak_async', return_value=False) as speak, \
             patch('voice.time.monotonic', return_value=10.):
            voice.announce('LOCKED')
            voice.announce('LOST')
            speak.return_value = True
            speak.reset_mock()
            voice.announce('LOST')
        speak.assert_called_once()
        self.assertIn(speak.call_args.args[0], voice._LOST_LINES)

    def test_manual_callouts_describe_visual_lock_without_claiming_hover(self):
        with patch('voice._speak_async', return_value=True) as speak, \
             patch('voice.time.monotonic', return_value=10.) as clock:
            voice.announce('REACHED', visual_only=True)
            text = speak.call_args.args[0].lower()
            self.assertNotIn('hover', text)
            self.assertNotIn('holding', text)
            self.assertNotIn('reached', text)
            speak.reset_mock()
            clock.return_value = 14.
            voice.announce('APPROACHING', visual_only=True)
            speak.assert_not_called()  # Still the same visual lock.
            voice.announce('LOST', visual_only=True)
            speak.assert_called_once_with('Visual target lost.')


if __name__ == '__main__':
    unittest.main()
