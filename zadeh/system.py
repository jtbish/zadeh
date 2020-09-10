class FuzzyRuleBasedSystem:
    def __init__(self, inference_engine, rule_base):
        self._inference_engine = inference_engine
        self._rule_base = rule_base

    @property
    def inference_engine(self):
        return self._inference_engine

    @property
    def rule_base(self):
        return self._rule_base

    def score(self, input_vec):
        return self._inference_engine.score(self._rule_base, input_vec)

    def classify(self, input_vec):
        return self._inference_engine.classify(self._rule_base, input_vec)
