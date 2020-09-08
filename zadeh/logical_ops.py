from .constants import MATCHING_MIN, MATCHING_MAX


def logical_or_max(membership_vals):
    return _operate_on_membership_vals(membership_vals, operator=max)


def logical_and_min(membership_vals):
    return _operate_on_membership_vals(membership_vals, operator=min)


def _operate_on_membership_vals(membership_vals, operator):
    assert len(membership_vals) > 0
    result = operator(membership_vals)
    assert MATCHING_MIN <= result <= MATCHING_MAX
    return result
