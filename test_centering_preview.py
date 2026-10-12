import unittest
from unittest.mock import patch

import config
from centering_preview import CenteringPreview
from controller import HOVER


def person(**changes):
    return dict(dict(found=True,x=.5,y=.5,w=.2,h=.4,conf=.9),**changes)


class CenteringTests(unittest.TestCase):
    def test_side_to_side_sequence_turns_toward_current_position(self):
        ctrl=CenteringPreview()
        for x,sign in ((.3,-1),(.4,-1),(.5,0),(.65,1),(.8,1),(.5,0),(.2,-1)):
            result=ctrl.update(person(x=x),enabled=True)
            yaw=result['command']['yaw']
            self.assertEqual((yaw>0)-(yaw<0),sign)
            self.assertLessEqual(abs(yaw),.15)
            for axis in ('forward','up','right'):
                self.assertEqual(result['command'][axis],0)

    def test_hysteresis_reduces_boundary_chatter_without_stale_direction(self):
        ctrl=CenteringPreview()
        self.assertEqual(ctrl.update(person(x=.545),enabled=True)['desired_yaw'],0)
        self.assertGreater(ctrl.update(person(x=.57),enabled=True)['desired_yaw'],0)
        self.assertGreater(ctrl.update(person(x=.545),enabled=True)['desired_yaw'],0)
        self.assertEqual(ctrl.update(person(x=.52),enabled=True)['desired_yaw'],0)
        self.assertEqual(ctrl.update(person(x=.545),enabled=True)['desired_yaw'],0)
        ctrl.update(person(x=.8),enabled=True)
        self.assertLess(ctrl.update(person(x=.2),enabled=True)['desired_yaw'],0)

    def test_off_loss_and_stale_video_immediately_clear_request(self):
        ctrl=CenteringPreview()
        cases=[(person(),False,True),(person(),True,False),
               (dict(found=False),True,True),(person(conf=.5),True,True),
               (person(x=float('nan')),True,True)]
        for target,enabled,fresh in cases:
            ctrl.update(person(x=.8),enabled=True)
            result=ctrl.update(target,enabled=enabled,fresh=fresh)
            self.assertEqual(result['command'],HOVER)
            self.assertEqual(result['desired_yaw'],0)
            self.assertFalse(ctrl.turning)
        self.assertEqual(CenteringPreview().update(person(x=.8))['command'],HOVER)

    def test_cropped_legs_allow_direction_diagnostic_but_never_bypass_stop(self):
        ctrl=CenteringPreview()
        for target in (person(x=.7,y=.63,h=.73),person(x=.7,w=.35),person(x=.7,h=.7)):
            result=ctrl.update(target,enabled=True)
            self.assertGreater(result['desired_yaw'],0)
            self.assertEqual(result['command'],HOVER)
            self.assertIn('stop',result['reason'])

    def test_lower_caps_are_respected(self):
        for cap in (.02,0,-.1):
            with patch.object(config,'MAX_YAW',cap):
                result=CenteringPreview().update(person(x=.8),enabled=True)
                self.assertLessEqual(abs(result['desired_yaw']),max(0,cap))

    def test_downward_view_does_not_assume_yaw_will_center_target(self):
        ctrl=CenteringPreview()
        ctrl.update(person(x=.8),enabled=True)
        result=ctrl.update(person(x=.8),enabled=True,view_supported=False)
        self.assertEqual(result['command'],HOVER)
        self.assertEqual(result['desired_yaw'],0)
        self.assertFalse(ctrl.turning)
        self.assertIn('yaw mapping',result['reason'])


if __name__=='__main__':
    unittest.main()
