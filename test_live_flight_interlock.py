"""Incident regressions. No drone, camera or real network connection is opened."""
import unittest
from unittest.mock import Mock, patch

import config
import drone
import main


class LiveFlightInterlockTests(unittest.TestCase):
    def test_unrestricted_live_modes_rejected_before_hardware_or_models(self):
        for mode in ({}, {'manual': True}, {'follow': True}):
            with self.subTest(mode=mode), \
                 patch.object(config, 'CONTROL_PORT', 8800), \
                 patch.object(main, 'VideoSource') as video, \
                 patch.object(main, 'DroneLink') as link, \
                 patch.object(main, 'PersonTracker') as tracker:
                with self.assertRaisesRegex(RuntimeError, 'disabled'):
                    main.main(**mode)
                video.assert_not_called()
                link.assert_not_called()
                tracker.assert_not_called()

    def test_restricted_modes_open_only_the_requested_control_type(self):
        for mode, expected in (({'person': True}, {'flight_test': False, 'yaw_only': True}),
                               ({'flight_test': True}, {'flight_test': True, 'yaw_only': False})):
            video, link = Mock(), Mock()
            video.kind = 'drone'
            video.read.side_effect = RuntimeError('end simulated loop')
            with self.subTest(mode=mode), \
                 patch.object(config, 'CONTROL_PORT', 8800), \
                 patch.object(config, 'VIDEO_SOURCE', 'drone'), \
                 patch.object(main, 'VideoSource', return_value=video), \
                 patch.object(main, 'DroneLink', return_value=link) as factory, \
                 patch.object(main, 'PersonTracker'), \
                 patch.object(main, 'AsyncPersonTracker'), \
                 patch.object(main.cv2, 'namedWindow'), \
                 patch.object(main.cv2, 'setMouseCallback'), \
                 patch.object(main.cv2, 'destroyAllWindows'):
                with self.assertRaisesRegex(RuntimeError, 'end simulated loop'):
                    main.main(**mode)
                factory.assert_called_once_with(**expected)
                link.takeoff.assert_not_called()
                link.send.assert_not_called()

    def test_manual_and_automatic_permissions_cannot_be_combined(self):
        with self.assertRaises(ValueError):
            drone.require_safe_live_control(flight_test=True, yaw_only=True)

    def test_live_person_rejects_non_drone_camera_before_hardware(self):
        with patch.object(config, 'CONTROL_PORT', 8800), \
             patch.object(config, 'VIDEO_SOURCE', 'webcam'), \
             patch.object(main, 'VideoSource') as video:
            with self.assertRaisesRegex(RuntimeError, 'drone video only'):
                main.main(person=True)
            video.assert_not_called()

    def test_direct_control_link_also_rejected_before_adapter_or_sender(self):
        with patch.object(config, 'CONTROL_PORT', 8800), \
             patch.object(drone, '_load_turbodrone') as load, \
             patch.object(drone.threading, 'Thread') as thread:
            with self.assertRaisesRegex(RuntimeError, 'unexpected takeoff'):
                drone.DroneLink()
            load.assert_not_called()
            thread.assert_not_called()

    def test_dry_run_still_available_without_adapter_or_sender(self):
        with patch.object(config, 'CONTROL_PORT', None), \
             patch.object(drone, '_load_turbodrone') as load, \
             patch.object(drone.threading, 'Thread') as thread:
            link = drone.DroneLink()
            link.takeoff()
            link.land()
            link.close()
            self.assertTrue(link.dry_run)
            load.assert_not_called()
            thread.assert_not_called()


if __name__ == '__main__':
    unittest.main()
