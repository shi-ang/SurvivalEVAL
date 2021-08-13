import numpy as np
import warnings

import lifelines.utils
class SurvivalEvaluator:
    def __init__(self):
        raise NotImplementedError

    def concordance(self, ties, method):
        raise NotImplementedError

    def brier_score(self, target_time):
        raise NotImplementedError

    def integrated_brier_score(self):
        raise NotImplementedError

    def one_calibration(self, target_time, bins):
        raise NotImplementedError

    def d_calibration(self, bins):
        raise NotImplementedError

lifelines.utils.qth_survival_time()