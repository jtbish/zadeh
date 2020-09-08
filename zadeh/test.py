import numpy as np
from piecewise.environment import make_mountain_car_test_env as mc

from aggregation import AvgMatchingAggregation
from antecedent import ConjunctiveAntecedent
from domain import Domain
from inference_engine import InferenceEngine
from linguistic_var import LinguisticVar
from logical_ops import logical_and_min
from membership_func import make_triangular_membership_func as make_tri
from rule import FuzzyRule
from rule_base import FuzzyRuleBase

NORMALISE_OBSS = False
NUM_POS_MEMBERSHIP_FUNCS = 5
NUM_VEL_MEMBERSHIP_FUNCS = 5


def main():
    ref_env = mc(seed=0, normalise=NORMALISE_OBSS)
    ling_vars = _make_ling_vars(ref_env.obs_space)
    class_labels = ref_env.action_set
    rule_base = _make_rule_base(class_labels)
    aggregation_strat = AvgMatchingAggregation()
    inference_engine = InferenceEngine(ling_vars,
                                       class_labels,
                                       rule_base,
                                       logical_and_strat=logical_and_min,
                                       aggregation_strat=aggregation_strat)

    resolution = 100
    poss = np.linspace(ref_env.obs_space[0].lower,
                       ref_env.obs_space[0].upper,
                       resolution,
                       endpoint=True)
    vels = np.linspace(ref_env.obs_space[1].lower,
                       ref_env.obs_space[1].upper,
                       resolution,
                       endpoint=True)
    test_points = list(zip(poss, vels))
    for test_point in test_points:
        score_array = inference_engine.score(test_point)
        class_label = inference_engine.classify(test_point)
        print(f"({test_point[0]:.3f}, {test_point[1]:.3f}) -> {score_array}, "
              f"{class_label}")


def _make_ling_vars(obs_space):
    assert NUM_POS_MEMBERSHIP_FUNCS >= 3 and (NUM_POS_MEMBERSHIP_FUNCS %
                                              2) == 1
    assert NUM_VEL_MEMBERSHIP_FUNCS >= 3 and (NUM_VEL_MEMBERSHIP_FUNCS %
                                              2) == 1
    ling_vars = []

    pos_domain = Domain(obs_space[0].lower, obs_space[0].upper)
    pos_mfs = _make_uniformly_spaced_tris(pos_domain, NUM_POS_MEMBERSHIP_FUNCS,
                                          "pos")
    ling_vars.append(LinguisticVar(pos_mfs, "pos"))

    vel_domain = Domain(obs_space[1].lower, obs_space[1].upper)
    vel_mfs = _make_uniformly_spaced_tris(vel_domain, NUM_VEL_MEMBERSHIP_FUNCS,
                                          "vel")
    ling_vars.append(LinguisticVar(vel_mfs, "vel"))

    return ling_vars


def _make_uniformly_spaced_tris(domain, num_membership_funcs, feature_name):
    ref_points = np.linspace(domain.min,
                             domain.max,
                             num=num_membership_funcs,
                             endpoint=True)
    membership_funcs = []
    # first mf
    membership_funcs.append(
        make_tri(domain, ref_points[0], ref_points[0], ref_points[1],
                 f"{feature_name}_0"))
    # middle mfs
    for i in range(0, (num_membership_funcs - 2)):
        membership_funcs.append(
            make_tri(domain, ref_points[i], ref_points[i + 1],
                     ref_points[i + 2], f"{feature_name}_{i+1}"))
    # last mf
    membership_funcs.append(
        make_tri(domain, ref_points[-2], ref_points[-1], ref_points[-1],
                 f"{feature_name}_{num_membership_funcs-1}"))
    return membership_funcs


def _make_rule_base(class_labels):
    rules = []
    for i in range(0, NUM_POS_MEMBERSHIP_FUNCS):
        for j in range(0, NUM_VEL_MEMBERSHIP_FUNCS):
            membership_func_idxs = [i, j]
            antecedent = ConjunctiveAntecedent(membership_func_idxs)
            consequent = np.random.choice(tuple(class_labels))
            rules.append(FuzzyRule(antecedent, consequent))
    rule_base = FuzzyRuleBase(rules, default_class_label=None)
    return rule_base


if __name__ == "__main__":
    main()
