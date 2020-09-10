from collections import OrderedDict, namedtuple

import numpy as np

from constants import MATCHING_MIN
from error import UndefinedMappingError

MatchingRecord = namedtuple("MatchingRecord", ["rule", "matching_degree"])


class InferenceEngine:
    """FITA inference engine."""
    def __init__(self,
                 ling_vars,
                 class_labels,
                 logical_and_strat=None,
                 logical_or_strat=None,
                 aggregation_strat=None):
        self._ling_vars = ling_vars
        self._class_labels = class_labels
        self._logical_and_strat = logical_and_strat
        self._logical_or_strat = logical_or_strat
        self._aggregation_strat = aggregation_strat

    def score(self, rule_base, input_vec):
        """Takes input vector of features, returns array of score values,
        one for each class."""
        matching_records = self._compute_matching_records(rule_base, input_vec)
        match_set = self._build_match_set(matching_records)
        score_array = self._aggregation_strat(match_set, self._class_labels)
        return score_array

    def _compute_matching_records(self, rule_base, input_vec):
        matching_records = []
        for rule in rule_base:
            matching_degree = rule.eval_antecedent(self._ling_vars, input_vec,
                                                   self._logical_and_strat,
                                                   self._logical_or_strat)
            matching_records.append(MatchingRecord(rule, matching_degree))
        return matching_records

    def _build_match_set(self, matching_records):
        return [
            matching_record for matching_record in matching_records
            if matching_record.matching_degree > MATCHING_MIN
        ]

    def classify(self, rule_base, input_vec):
        score_array = self.score(rule_base, input_vec)
        filtered_score_array = \
            self._filter_nan_entries_from_score_array(score_array)
        has_at_least_one_score = len(filtered_score_array) > 0
        if has_at_least_one_score:
            return max(filtered_score_array, key=filtered_score_array.get)
        else:
            return self._use_default_class_label(rule_base)

    def _filter_nan_entries_from_score_array(self, score_array):
        return OrderedDict({
            class_label: score
            for (class_label, score) in score_array.items()
            if (not np.isnan(score))
        })

    def _use_default_class_label(self, rule_base):
        if rule_base.has_default_class_label():
            return rule_base.default_class_label
        else:
            raise UndefinedMappingError