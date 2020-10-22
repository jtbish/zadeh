import numpy as np

from .constants import FRBS_COMPLEXITY_MAX, FRBS_COMPLEXITY_MIN


class FuzzyRuleBasedSystem:
    def __init__(self, inference_engine, ling_vars, rule_base):
        self._inference_engine = inference_engine
        self._ling_vars = ling_vars
        self._rule_base = rule_base

    @property
    def inference_engine(self):
        return self._inference_engine

    @property
    def ling_vars(self):
        return self._ling_vars

    @property
    def rule_base(self):
        return self._rule_base

    def score(self, input_vec):
        return self._inference_engine.score(self._ling_vars, self._rule_base,
                                            input_vec)

    def classify(self, input_vec):
        return self._inference_engine.classify(self._ling_vars,
                                               self._rule_base, input_vec)

    def calc_complexity(self):
        num_fuzzy_decision_regions = \
            np.prod([lv.num_membership_funcs for lv in self._ling_vars])
        num_spec_fdrs = self._rule_base.num_spec_fuzzy_decision_regions()
        complexity = num_spec_fdrs/num_fuzzy_decision_regions
        assert FRBS_COMPLEXITY_MIN <= complexity <= FRBS_COMPLEXITY_MAX
        return complexity
