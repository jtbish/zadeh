class FuzzyRuleBase:
    def __init__(self, rules):
        self._rules = tuple(rules)

    def num_spec_fuzzy_decision_regions(self):
        return sum(
            [rule.num_spec_fuzzy_decision_regions() for rule in self._rules])

    def __iter__(self):
        return iter(self._rules)

    def __len__(self):
        return len(self._rules)
