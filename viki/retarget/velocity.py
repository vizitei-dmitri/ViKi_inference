"""Causal velocity from distinct source observations, with bounded persistence."""

import numpy as np

from viki.contracts import LiveIKTuning


class CausalTargetVelocity:
    """No extrapolated positions, timestamp renewal or updates on held samples."""

    def __init__(self, tuning: LiveIKTuning):
        tuning.validate()
        self.tuning = tuning
        self.reset()

    def reset(self):
        self.position = None
        self.source_us = None
        self.arrival_us = None
        self.now_us = None
        self.velocity = np.zeros(3)

    def update(self, position, source_us, now_us):
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("velocity target must be a finite XYZ position")
        if source_us > now_us or (self.now_us is not None and now_us < self.now_us):
            raise ValueError("invalid velocity clock chronology")
        if self.source_us is not None and source_us < self.source_us:
            raise ValueError("velocity source time went backwards")
        if self.source_us == source_us and not np.array_equal(position, self.position):
            raise ValueError("changed velocity input at the same source time")
        self.now_us = now_us
        if self.source_us != source_us:
            dt = (source_us-self.source_us)/1e6 if self.source_us is not None else None
            if dt is None or dt > .25:
                self.velocity[:] = 0.
            else:
                raw = (position-self.position)/dt
                speed = np.linalg.norm(raw)
                raw *= min(1., self.tuning.feedforward_max_speed/max(speed, 1e-12))
                alpha = 1/(1+1/(2*np.pi*self.tuning.feedforward_cutoff_hz*dt))
                self.velocity += alpha*(raw-self.velocity)
            self.position, self.source_us, self.arrival_us = position.copy(), source_us, now_us
        age = (now_us-self.arrival_us)/1e6
        hold = self.tuning.feedforward_hold_s
        # Fade in the second half of the hold interval; reading a held target
        # never restarts the interval. The live input gate separately enforces TTL.
        fade = np.clip(2*(1-age/hold), 0., 1.)
        return self.velocity.copy()*fade
