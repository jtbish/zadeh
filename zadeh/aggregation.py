import abc
from collections import OrderedDict

import numpy as np

from constants import MATCHING_MAX, MATCHING_MIN


def _init_score_array(class_labels):
    return OrderedDict({class_label: np.nan for class_label in class_labels})


class AggregationStrategyABC(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, matching_records, class_labels):
        raise NotImplementedError


class AvgMatchingAggregation(AggregationStrategyABC):
    def __call__(self, matching_records, class_labels):
        score_array = _init_score_array(class_labels)

        matching_sums = {
            class_label: MATCHING_MIN
            for class_label in class_labels
        }
        advocation_counts = {class_label: 0 for class_label in class_labels}
        for matching_record in matching_records:
            class_label = matching_record.rule.consequent
            matching_sums[class_label] += matching_record.matching_degree
            advocation_counts[class_label] += 1

        for class_label in class_labels:
            advocation_count = advocation_counts[class_label]
            if advocation_count > 0:
                score = (matching_sums[class_label] / advocation_count)
                score_array[class_label] = score

        return score_array


class MatchingSumAggregation(AggregationStrategyABC):
    def __call__(self, matching_records, class_labels):
        score_array = _init_score_array(class_labels)

        matching_sums = {
            class_label: MATCHING_MIN
            for class_label in class_labels
        }
        advocation_counts = {class_label: 0 for class_label in class_labels}
        for matching_record in matching_records:
            class_label = matching_record.rule.consequent
            matching_sums[class_label] += matching_record.matching_degree
            advocation_counts[class_label] += 1

        for class_label in class_labels:
            advocation_count = advocation_counts[class_label]
            if advocation_count > 0:
                raw_matching_sum = matching_sums[class_label]
                bounded_matching_sum = min(MATCHING_MAX, raw_matching_sum)
                score_array[class_label] = bounded_matching_sum

        return score_array
