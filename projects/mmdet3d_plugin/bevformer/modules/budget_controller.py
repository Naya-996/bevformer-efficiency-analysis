"""Deterministic hysteresis controller for budget-adaptive BEV resolution."""

import time


class BudgetAdaptiveResolutionController:
    """Select one of ordered resolutions using cheap previous-frame signals."""

    def __init__(self, resolutions=(100, 150, 200), safe_resolution=200,
                 up_threshold=0.65, down_threshold=0.35, min_residency=3,
                 translation_scale=2.0, yaw_scale=10.0,
                 translation_weight=0.35, yaw_weight=0.25,
                 low_confidence_weight=0.40):
        self.resolutions = tuple(sorted(int(value) for value in resolutions))
        if int(safe_resolution) not in self.resolutions:
            raise ValueError('safe_resolution must be present in resolutions')
        if down_threshold >= up_threshold:
            raise ValueError('down_threshold must be lower than up_threshold')
        self.safe_resolution = int(safe_resolution)
        self.up_threshold = float(up_threshold)
        self.down_threshold = float(down_threshold)
        self.min_residency = int(min_residency)
        self.translation_scale = float(translation_scale)
        self.yaw_scale = float(yaw_scale)
        self.weights = (
            float(translation_weight), float(yaw_weight),
            float(low_confidence_weight))
        self.reset()

    def reset(self):
        self.current_resolution = self.safe_resolution
        self.resident_frames = 0
        self.last_record = None

    @staticmethod
    def _clamp01(value):
        return max(0.0, min(1.0, float(value)))

    def score(self, signals):
        translation = self._clamp01(
            signals.get('translation_m', 0.0) / self.translation_scale)
        yaw = self._clamp01(abs(signals.get('yaw_deg', 0.0)) / self.yaw_scale)
        low_confidence = self._clamp01(signals.get('low_confidence_ratio', 0.0))
        components = (translation, yaw, low_confidence)
        normalizer = sum(self.weights)
        return sum(w * value for w, value in zip(self.weights, components)) / normalizer

    def select(self, signals=None, scene_changed=False):
        started = time.perf_counter()
        signals = signals or {}
        previous = self.current_resolution
        difficulty = self.score(signals)
        reason = 'residency_or_hysteresis'
        if scene_changed:
            self.reset()
            reason = 'scene_reset'
        elif self.resident_frames >= self.min_residency:
            index = self.resolutions.index(self.current_resolution)
            if difficulty >= self.up_threshold and index < len(self.resolutions) - 1:
                self.current_resolution = self.resolutions[index + 1]
                self.resident_frames = 0
                reason = 'difficulty_up'
            elif difficulty <= self.down_threshold and index > 0:
                self.current_resolution = self.resolutions[index - 1]
                self.resident_frames = 0
                reason = 'difficulty_down'
        self.resident_frames += 1
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.last_record = {
            'previous_resolution': previous,
            'resolution': self.current_resolution,
            'difficulty': difficulty,
            'reason': reason,
            'resident_frames': self.resident_frames,
            'controller_ms': elapsed_ms,
        }
        return self.current_resolution

