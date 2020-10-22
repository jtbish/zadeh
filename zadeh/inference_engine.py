from collections import namedtuple

import numpy as np

from .constants import SCORE_MAX, SCORE_MIN
from .error import UndefinedMappingError

MatchingRecord = namedtuple("MatchingRecord", ["rule", "matching_degree"])


class InferenceEngine:
    """FITA inference engine."""
    def __init__(self,
                 class_labels,
                 logical_and_strat=None,
                 logical_or_strat=None,
                 aggregation_strat=None):
        self._class_labels = class_labels
        self._logical_and_strat = logical_and_strat
        self._logical_or_strat = logical_or_strat
        self._aggregation_strat = aggregation_strat

    @property
    def class_labels(self):
        return self._class_labels

    def score(self, ling_vars, rule_base, input_vec):
        """Takes input vector of features, returns array of score values,
        one for each class."""
        matching_records = self._compute_matching_records(
            ling_vars, rule_base, input_vec)
        score_array = self._aggregation_strat(matching_records,
                                              self._class_labels)
        assert self._score_array_is_valid(score_array)
        return score_array

    def _compute_matching_records(self, ling_vars, rule_base, input_vec):
        matching_records = []
        for rule in rule_base:
            matching_degree = rule.eval_antecedent(ling_vars, input_vec,
                                                   self._logical_and_strat,
                                                   self._logical_or_strat)
            matching_records.append(MatchingRecord(rule, matching_degree))
        return matching_records

    def _score_array_is_valid(self, score_array):
        return np.all([
            SCORE_MIN <= score <= SCORE_MAX for score in score_array.values()
        ])

    def classify(self, ling_vars, rule_base, input_vec):
        score_array = self.score(ling_vars, rule_base, input_vec)
        if not self._all_scores_are_min(score_array):
            return max(score_array, key=score_array.get)
        else:
            raise UndefinedMappingError

    def _all_scores_are_min(self, score_array):
        return np.all([score == SCORE_MIN for score in score_array.values()])
