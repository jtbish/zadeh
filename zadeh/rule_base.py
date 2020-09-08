class RuleBase:
    def __init__(self, rules, default_class_label=None):
        self._rules = tuple(rules)
        self._default_class_label = default_class_label

    @property
    def default_class_label(self):
        return self._default_class_label
