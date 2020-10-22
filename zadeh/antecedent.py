import abc

import numpy as np

UNSPECIFIED = -1


class AntecedentABC(metaclass=abc.ABCMeta):
    """Antecedent stores a mapping between input features and linguistic values
    selected for those features."""
    @abc.abstractmethod
    def eval(self,
             ling_vars,
             input_vec,
             logical_and_strat,
             logical_or_strat=None):
        raise NotImplementedError

    @abc.abstractmethod
    def num_spec_fuzzy_decision_regions(self):
        raise NotImplementedError


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

    def num_spec_fuzzy_decision_regions(self):
        raise NotImplementedError


class CNFAntecedent(AntecedentABC):
    _ACTIVE = 1
    _INACTIVE = 0

    def __init__(self, membership_func_usages):
        for mf_usage_bits in membership_func_usages:
            for bit in mf_usage_bits:
                assert bit == self._ACTIVE or bit == self._INACTIVE
            at_least_one_active_bit = mf_usage_bits.count(self._ACTIVE) > 0
            assert at_least_one_active_bit
        self._membership_func_usages = tuple(membership_func_usages)

    def eval(self, ling_vars, input_vec, logical_and_strat, logical_or_strat):
        vals_to_and = []
        for (mf_usage_bits, ling_var, input_scalar) in \
                zip(self._membership_func_usages, ling_vars, input_vec):
            all_bits_active = \
                mf_usage_bits.count(self._ACTIVE) == len(mf_usage_bits)
            if all_bits_active:
                continue  # generalises over feature, don't compute anything
            else:
                vals_to_and.append(
                    self._eval_disjunction(mf_usage_bits, ling_var,
                                           input_scalar, logical_or_strat))
        result = logical_and_strat(vals_to_and)
        #print(f"Conjunction: {vals_to_and} -> {result}")
        return logical_and_strat(vals_to_and)

    def _eval_disjunction(self, mf_usage_bits, ling_var, input_scalar,
                          logical_or_strat):
        vals_to_or = []
        for (mf_idx, bit) in enumerate(mf_usage_bits):
            if bit == self._ACTIVE:
                vals_to_or.append(
                    ling_var.eval_membership_func(mf_idx, input_scalar))
        result = logical_or_strat(vals_to_or)
        #print(f"Disjunction: {mf_usage_bits} -> {vals_to_or} -> {result}")
        return logical_or_strat(vals_to_or)

    def num_spec_fuzzy_decision_regions(self):
        active_bits_per_ling_var = [
            mf_usage_bits.count(self._ACTIVE)
            for mf_usage_bits in self._membership_func_usages
        ]
        return np.prod(active_bits_per_ling_var)
