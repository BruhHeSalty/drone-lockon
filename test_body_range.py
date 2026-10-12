import unittest

from body_range import torso_measurement, approximate_range_feet


def points():
    p = [[0., 0., 0.] for _ in range(17)]
    for i, xy in zip((5, 6, 11, 12), ((.4, .2), (.6, .2), (.42, .6), (.58, .6))):
        p[i] = [*xy, .95]
    return p


class BodyRangeTests(unittest.TestCase):
    def test_missing_feet_do_not_prevent_torso_measurement(self):
        m = torso_measurement(points())
        self.assertAlmostEqual(m['torso_height'], .4)
        self.assertAlmostEqual(approximate_range_feet(m, .2, 12), 6)

    def test_weak_or_offscreen_torso_cannot_supply_range(self):
        for index in (5, 6, 11, 12):
            for bad in ([.4, .2, .5], [.4, .995, .99], [float('nan'), .2, .99]):
                p = points()
                p[index] = bad
                self.assertIsNone(torso_measurement(p))

    def test_no_guess_when_measurement_is_missing(self):
        self.assertIsNone(approximate_range_feet(None, .2, 12))
        self.assertIsNone(approximate_range_feet(torso_measurement(points()), 0, 12))

    def test_inverted_or_tiny_torso_is_rejected(self):
        p = points()
        p[11][1] = p[12][1] = .21
        self.assertIsNone(torso_measurement(p))


if __name__ == '__main__':
    unittest.main()
