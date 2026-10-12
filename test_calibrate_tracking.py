import unittest

from calibrate_tracking import measure


def row(seconds, found=True, **changes):
    target = dict(found=True, x=.5, y=.5, w=.2, h=.4, conf=.9)
    target.update(changes)
    return dict(seconds=seconds, found=found, selected_id=7, tracks={'7': target})


class CalibrationTests(unittest.TestCase):
    def test_unconfirmed_and_cropped_boxes_do_not_calibrate_full_body_size(self):
        result = measure([row(0), row(1, found=False), row(2, y=.95), row(3, conf=.2)], 0, 4)
        self.assertEqual(result['confirmed_frames'], 2)
        self.assertEqual(result['complete_body_frames'], 1)
        self.assertEqual(result['complete_body_box_fractions']['h']['median'], .4)
        self.assertEqual(result['status'], 'DISTANCE_NOT_MEASURED')

    def test_close_rule_and_time_window_use_actual_controller(self):
        result = measure([row(0), row(1, h=.7), row(2, w=.35), row(3)], 1, 3, 10)
        self.assertEqual(result['frames'], 2)
        self.assertEqual(result['close_stop_frames'], 2)
        self.assertEqual(result['user_supplied_measured_distance_feet'], 10)

    def test_no_detection_does_not_fabricate_measurements(self):
        result = measure([row(0, found=False)], 0, 1)
        self.assertIsNone(result['complete_body_box_fractions'])

    def test_estimate_is_not_promoted_to_measured_distance(self):
        result = measure([row(0)], 0, 1, 12, 'estimated')
        self.assertEqual(result['status'], 'ESTIMATED_REFERENCE_ONLY')
        self.assertEqual(result['user_supplied_distance_feet'], 12)
        self.assertIsNone(result['user_supplied_measured_distance_feet'])

    def test_invalid_distance_or_interval_is_rejected(self):
        for start, end, distance in [(0, 1, -1), (0, 1, float('nan')), (2, 1, None), (0, float('inf'), None), (5, 6, None)]:
            with self.subTest(start=start, end=end, distance=distance), self.assertRaises(ValueError):
                measure([row(0)], start, end, distance)


if __name__ == '__main__':
    unittest.main()
