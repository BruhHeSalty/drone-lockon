"""Offline horizontal-centering prototype. Not connected to main.py or a drone.

Keep the desired turn separate from the permitted command. A cropped person's
horizontal position can still be useful, but it does not establish clearance.
Video replay tests direction decisions, not physical tracking performance.
"""
import math

import config
from controller import Controller, HOVER


class CenteringPreview:
    START_ERROR = .06
    STOP_ERROR = .03
    YAW_CAP = .15

    def __init__(self):
        self.turning = False

    def update(self, target, enabled=False, fresh=True, view_supported=True):
        """Return diagnostics plus an unchanged-format Command.

        A smaller stop band avoids toggling at the start boundary. The sign is
        always taken from the current position, never carried over after loss.
        Full-body visibility is not required to compute horizontal error.
        The existing image-size stop still overrides ALL movement requests.
        """
        output = dict(command=dict(HOVER), desired_yaw=0., error=None,
                      reason='', direction='CENTERED')
        if not enabled or not fresh or not view_supported or not Controller._valid_person(target):
            self.turning = False
            output['direction'] = 'HOLD'
            output['reason'] = ('controls off' if not enabled else
                                'video stale' if not fresh else
                                'downward view: yaw mapping unverified' if not view_supported else
                                'target unconfirmed')
            return output
        error = target['x'] - .5
        output['error'] = error
        if abs(error) <= self.STOP_ERROR:
            self.turning = False
        elif abs(error) >= self.START_ERROR:
            self.turning = True
        gain = config.YAW_GAIN if math.isfinite(config.YAW_GAIN) else 0.
        cap = config.MAX_YAW if math.isfinite(config.MAX_YAW) else 0.
        magnitude = min(max(0., cap), self.YAW_CAP,
                        max(0., abs(error)-self.STOP_ERROR)*2*max(0.,gain)) if self.turning else 0.
        yaw = math.copysign(magnitude,error) if magnitude else 0.
        output['desired_yaw'] = yaw
        output['direction'] = 'TURN LEFT' if yaw < 0 else 'TURN RIGHT' if yaw > 0 else 'CENTERED'
        if target['w'] >= config.STOP_SIZE or target['h'] >= .65:
            output['reason'] = 'image-size stop: turn request held'
        else:
            output['command']['yaw'] = yaw
            output['reason'] = 'turn toward selected person' if yaw else 'inside center band'
        return output
