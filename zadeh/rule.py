class FuzzyRule:
    def __init__(self, antecedent, consequent):
        self._antecedent = antecedent
        self._consequent = consequent

    @property
    def consequent(self):
        return self._consequent

    def eval_antecedent(self,
                        ling_vars,
                        input_vec,
                        logical_and_strat=None,
                        logical_or_strat=None):
        return self._antecedent.eval(ling_vars, input_vec, logical_and_strat,
                                     logical_or_strat)

    def calc_num_spec_fuzzy_decision_regions(self):
        return self._antecedent.calc_num_spec_fuzzy_decision_regions()
