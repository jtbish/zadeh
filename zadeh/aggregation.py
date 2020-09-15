import abc
from collections import OrderedDict

from .constants import CONSEQUENT_MIN, CONSEQUENT_MAX, SCORE_MIN


def _init_score_array(class_labels):
    return OrderedDict({class_label: SCORE_MIN for class_label in class_labels})


def _implication(matching_record, class_label):
    """Product implication."""
    consequent_for_class = \
        matching_record.rule.consequent[class_label]
    assert CONSEQUENT_MIN <= consequent_for_class <= CONSEQUENT_MAX
    return matching_record.matching_degree * consequent_for_class


class AggregationStrategyABC(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, matching_records, class_labels):
        raise NotImplementedError


class MaximumAggregation(AggregationStrategyABC):
    def __call__(self, matching_records, class_labels):
        score_array = _init_score_array(class_labels)

        for class_labels in class_labels:
            implication_vals = [_implication(matching_record, class_label) for
                    matching_record in matching_records]
            score_array[class_label] = max(implication_vals)

        return score_array


class WeightedAvgAggregation(AggregationStrategyABC):
    def __call__(self, matching_records, class_labels):
        score_array = _init_score_array(class_labels)

        denominator = sum([matching_record.matching_degree for matching_record in
            matching_records])
        if denominator != 0.0:
            for class_labels in class_labels:
                numerator = sum([_implication(matching_record, class_label) for
                    matching_record in matching_records])
                score_array[class_label] = (numerator/denominator)

        return score_array
