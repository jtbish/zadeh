import numpy as np

from .constants import FLOAT_TOL, MATCHING_MAX, MATCHING_MIN
from .util import trunc_val


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
    assert (MATCHING_MIN - FLOAT_TOL) <= result <= (MATCHING_MAX + FLOAT_TOL)
    result = trunc_val(result, MATCHING_MIN, MATCHING_MAX)
    return result


def _probor(membership_vals):
    only_one_val = len(membership_vals) == 1
    if only_one_val:
        return membership_vals[0]
    else:
        return sum(membership_vals) - np.prod(membership_vals)
