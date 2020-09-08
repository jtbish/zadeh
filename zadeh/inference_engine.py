from collections import OrderedDict, namedtuple

import numpy as np

from .constants import RANGE_MAX, RANGE_MIN

MatchingRecord = namedtuple("MatchingRecord", ["rule", "matching_degree"])
MIN_MATCHING_DEGREE = RANGE_MIN
MAX_MATCHING_DEGREE = RANGE_MAX


class InferenceEngine:
    """FITA inference engine."""
    def __init__(self,
                 ling_vars,
                 class_labels,
                 rule_base,
                 logical_and_strat=None,
                 logical_or_strat=None,
                 aggregation_strat=None):
        self._ling_vars = ling_vars
        self._class_labels = class_labels
        self._rule_base = rule_base
        self._logical_and_strat = logical_and_strat
        self._logical_or_strat = logical_or_strat
        self._aggregation_strat = aggregation_strat

    def score(self, input_vec):
        """Takes input vector of features, returns array of score values,
        one for each class."""
        matching_records = self._compute_matching_records(input_vec)
        score_array = self._aggregation_strat(matching_records,
                                              self._class_labels)
        return score_array

    def _compute_matching_records(self, input_vec):
        matching_records = []
        for rule in self._rule_base:
            matching_degree = rule.eval_antecedent(self._ling_vars, input_vec,
                                                   self._logical_and_strat,
                                                   self._logical_or_strat)
            matching_records.append(MatchingRecord(rule, matching_degree))
        return tuple(matching_records)

    def classify(self, input_vec):
        score_array = self.score(input_vec)
        filtered_score_array = \
            self._filter_nan_entries_from_score_array(score_array)
        has_valid_scores = len(filtered_score_array > 0)
        if has_valid_scores:
            return max(filtered_score_array, key=filtered_score_array.get)
        else:
            return self._try_default_class_label()

    def _filter_nan_entries_from_score_array(self, score_array):
        return OrderedDict({
            class_label: score
            for (class_label, score) in score_array.items()
            if (not np.isnan(score))
        })

    def _try_default_class_label(self):
        default_class_label = self._rule_base.default_class_label
        if default_class_label is None:
            # TODO error
            assert False
        else:
            return default_class_label
