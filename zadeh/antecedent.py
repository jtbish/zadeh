import abc

UNSPECIFIED = -1


class AntecedentABC(metaclass=abc.ABCMeta):
    """Antecedent stores a mapping between input features and linguistic values
    selected for those features."""
    pass


class ConjunctiveAntecedent(AntecedentABC):
    def __init__(self, membership_func_idxs):
        self._membership_func_idxs = tuple(membership_func_idxs)

    def eval(self,
             ling_vars,
             input_vec,
             logical_and_strat,
             logical_or_strat=None):
        membership_vals = []
        for (membership_func_idx, ling_var,
             input_scalar) in zip(self._membership_func_idxs, ling_vars,
                                  input_vec):
            if membership_func_idx != UNSPECIFIED:
                membership_vals.append(
                    ling_var.eval_membership_func(membership_func_idx,
                                                  input_scalar))
        return logical_and_strat(membership_vals)
