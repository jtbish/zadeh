class FuzzyRuleBase:
    def __init__(self, rules):
        self._rules = tuple(rules)

    def calc_num_spec_fuzzy_decision_regions(self):
        return sum(
            [rule.calc_num_spec_fuzzy_decision_regions()
                for rule in self._rules])

    def __iter__(self):
        return iter(self._rules)

    def __len__(self):
        return len(self._rules)

    def __repr__(self):
        return str(self)

    def __str__(self):
        str_ = ""
        for rule in self._rules:
            str_ += f"{str(rule.antecedent)} -> {str(rule.consequent)}\n"
        return str_
