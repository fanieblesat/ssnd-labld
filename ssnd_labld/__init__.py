"""
ssnd_labld -- reference implementation of the model and the bundled Lagrangian
decomposition described in the draft

    "Learning-Augmented Lagrangian Decomposition for Sustainable and Resilient
     Multimodal Freight Service Network Design under Uncertainty"

Modules
-------
instance     data structures (SSNDInstance, Arc, Commodity, Scenario)
generator    synthetic multimodal time-space instances with demand + disruption scenarios
models       PuLP models: recourse LP, deterministic equivalent MILP, bundle subproblem MILP
bundling     scenario partitions: singleton, random, demand/disruption k-means, solution-similarity,
             cost-signature (k-means on recourse costs at candidate designs)
warmstart    dual warm starts: zero, LP-dual, external (e.g. a learned predictor)
lagrangian   bundled Lagrangian decomposition with projected subgradient, guard, primal recovery
learning     learned dual warm start (MultiplierPredictor) and learned partition selection
             (PartitionScorer); both are safe: guard / choice among valid partitions (v5)

Conventions (consistent with AUDIT_REPORT.md, section 4.7)
----------------------------------------------------------
* Nonanticipativity across bundles is dualised with one multiplier vector per
  bundle, lambda_b in R^{|A_S|}, in the *balanced* space
      Lambda = { (lambda_1..lambda_m) : sum_b lambda_b = 0 }.
* Projection onto Lambda uses the UNWEIGHTED mean:
      Pi(lambda)_b = lambda_b - (1/m) sum_c lambda_c.
* The projected subgradient of the dual function L at lambda is therefore
      g_b = y^b - (1/m) sum_c y^c,
  NOT the probability-weighted PH centring y^b - sum_c P_c y^c (that belongs to
  the weighted representation lambda_b = P_b w_b, which is an equivalent
  reparameterisation but must not be mixed with the unweighted projection).
* Every evaluated L(lambda) with lambda in Lambda is a valid lower bound on the
  optimum of the deterministic equivalent (Proposition 1); the example script
  checks this numerically.
"""
from .instance import Arc, Commodity, Scenario, SSNDInstance
from .generator import generate_instance
from .models import (solve_recourse, evaluate_design, solve_deterministic_equivalent,
                     solve_bundle_subproblem, get_solver)
from .bundling import (singleton_bundles, random_bundles, kmeans_bundles,
                       solution_similarity_bundles, scenario_features,
                       cost_signature_bundles, cost_signatures, stratified_bundles)
from .warmstart import zero_multipliers, lp_dual_multipliers, project_balanced
from .lagrangian import BundledLagrangian, LDResult
from .models import solve_bundle_lp
from .learning import (MultiplierPredictor, PartitionScorer, candidate_partitions, partition_features,
                       multiplier_features, lp_proxy_scores, save_models, load_models)

__all__ = [
    "Arc", "Commodity", "Scenario", "SSNDInstance", "generate_instance",
    "solve_recourse", "evaluate_design", "solve_deterministic_equivalent",
    "solve_bundle_subproblem", "get_solver",
    "singleton_bundles", "random_bundles", "kmeans_bundles",
    "solution_similarity_bundles", "scenario_features",
    "cost_signature_bundles", "cost_signatures", "stratified_bundles",
    "zero_multipliers", "lp_dual_multipliers", "project_balanced",
    "BundledLagrangian", "LDResult", "solve_bundle_lp",
    "MultiplierPredictor", "PartitionScorer", "candidate_partitions", "partition_features",
    "multiplier_features", "lp_proxy_scores", "save_models", "load_models",
]
