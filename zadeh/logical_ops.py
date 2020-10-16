import numpy as np

from .constants import MATCHING_MAX, MATCHING_MIN


def logical_or_max(membership_vals):
    return _operate_on_membership_vals(membership_vals, operator=max)


def logical_or_probor(membership_vals):
    return _operate_on_membership_vals(membership_vals, operator=_probor)


def logical_and_min(membership_vals):
    return _operate_on_membership_vals(membership_vals, operator=min)


def logical_and_prod(membership_vals):
    return _operate_on_membership_vals(membership_vals, operator=np.prod)


def _operate_on_membership_vals(membership_vals, operator):
    assert len(membership_vals) > 0
    result = operator(membership_vals)
    assert MATCHING_MIN <= result <= MATCHING_MAX
    return result


def _probor(membership_vals):
    return sum(membership_vals) - np.prod(membership_vals)
