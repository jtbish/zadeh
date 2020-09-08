class FuzzyRuleBase:
    def __init__(self, rules, default_class_label=None):
        self._rules = tuple(rules)
        self._default_class_label = default_class_label

    @property
    def default_class_label(self):
        return self._default_class_label

    def has_default_class_label(self):
        return self._default_class_label is not None

    def __iter__(self):
        return iter(self._rules)
