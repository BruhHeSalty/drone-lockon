"""Incident regressions. No drone, camera or real network connection is opened."""
import unittest
from unittest.mock import patch

import config
import drone
import main


class LiveFlightInterlockTests(unittest.TestCase):
    def test_all_live_entry_modes_rejected_before_hardware_or_models(self):
        for mode in ({}, {'person': True}, {'manual': True}, {'flight_test': True}, {'follow': True}):
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
