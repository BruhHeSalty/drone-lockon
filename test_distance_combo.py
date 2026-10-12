import math
import unittest

from distance_combo import size_depth, ground_depth, combine_estimates, ground_contact


class ComboTests(unittest.TestCase):
    def test_size_ratio_and_missing_measurement(self):
        self.assertAlmostEqual(size_depth(.4, .1, 12), 3)
        for size in (None, 0, -1, float('nan')):
            self.assertIsNone(size_depth(size, .1, 12))

    def test_level_camera_projects_same_depth_at_feet_and_torso(self):
        cal = dict(normal=[0,1,0], offset_feet=5, fx_norm=1, fy_norm=1,
                   pose_verified=True, valid_from=0, valid_until=10)
        depth, reason = ground_depth((.5,.9), (.5,.6), cal, 1)
        self.assertIsNone(reason)
        self.assertAlmostEqual(depth, 12.5)

    def test_tilted_camera_does_not_mix_foot_depth_with_body_depth(self):
        s,c = math.sin(math.pi/4),math.cos(math.pi/4)
        # Foot=(0,5,10), normal=(0,c,s); body is 4ft above the foot.
        foot=[0,5,10]; body=[0,5-4*c,10-4*s]
        cal=dict(normal=[0,c,s],offset_feet=15*c,fx_norm=1,fy_norm=1,
                 pose_verified=True,valid_from=0,valid_until=1)
        depth,reason=ground_depth((.5,1),(.5,.5+body[1]/body[2]),cal,0)
        self.assertIsNone(reason)
        self.assertAlmostEqual(depth,body[2])
        self.assertNotAlmostEqual(depth,foot[2])

    def test_unknown_expired_or_unverified_ground_is_not_guessed(self):
        cal=dict(normal=[0,1,0],offset_feet=5,fx_norm=1,fy_norm=1,
                 pose_verified=True,valid_from=0,valid_until=1)
        for calibration,time in [(None,0),(dict(cal,pose_verified=False),0),(cal,2)]:
            self.assertIsNone(ground_depth((.5,.9),(.5,.6),calibration,time)[0])
        self.assertIsNone(ground_depth((.5,.501),(.5,.4),cal,0)[0])

    def test_conflict_loss_and_incompatible_points_do_not_produce_distance(self):
        for result in (combine_estimates(False,12,12), combine_estimates(True,12,4),
                       combine_estimates(True,12,None,4,'face'),
                       combine_estimates(True,12,None,12,'feet')):
            self.assertIsNone(result['feet'])
        self.assertEqual(combine_estimates(True,None,8)['source'],'torso')
        self.assertEqual(combine_estimates(True,8,8,8,'face')['source'],'face + ground check')

    def test_cropped_feet_do_not_become_ground_contact(self):
        p=[[.5,.9,.99] for _ in range(17)]
        t=dict(found=True,y=.6,h=.7)
        self.assertIsNotNone(ground_contact(t,p))
        self.assertIsNone(ground_contact(dict(t,y=.7),p))
        p[15][2]=.2
        self.assertIsNone(ground_contact(t,p))


if __name__ == '__main__':
    unittest.main()
