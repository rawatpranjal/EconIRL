RDW Vehicle Scrappage
=====================

.. image:: /_static/rdw_scrappage_overview.png
   :alt: RDW vehicle scrappage model showing scrappage probability surface over age and defect severity, sample vehicle trajectories, and estimated value function.
   :width: 100%

Every year in the Netherlands each passenger vehicle over three years old must pass a mandatory APK roadworthiness inspection. The vehicle owner observes the car's age and the severity of any defects found during the inspection. The owner then decides whether to keep the car and pay for repairs or to scrap the vehicle and exit the fleet. This example formulates the scrappage decision as a dynamic discrete choice model in the tradition of Rust (1987) and estimates it on real administrative data from the Dutch RDW open data registry.

The dataset contains 170,852 Volkswagen Golf vehicles registered between 2005 and 2015, observed annually from age 3 through either scrappage or right-censoring at the current date. The 2,009,144 vehicle-year observations include 17,106 exit events (scrappage or export). Defect severity comes from 447,296 individual defect records downloaded from the RDW geconstateerde gebreken dataset. The scrappage hazard rises from near zero at age 3 to 8.2 percent at age 20, and major defects (structural rust, braking failures) roughly double the exit probability conditional on age.

The structural model
--------------------

The owner of vehicle :math:`i` in period :math:`t` observes the state :math:`s_{it} = (\text{age}_{it}, d_{it})` where :math:`\text{age}_{it} \in \{0, 1, \ldots, 24\}` is the vehicle age in years and :math:`d_{it} \in \{0, 1, 2\}` is the APK defect severity (0 for pass, 1 for minor, 2 for major). The owner chooses action :math:`a_{it} \in \{0, 1\}` where 0 means keep and 1 means scrap. The per-period flow utility is

.. math::

   u(s, a; \theta) = \begin{cases}
   -\theta_1 \cdot \text{age} - \theta_2 \cdot \mathbf{1}[d = 1] - \theta_3 \cdot \mathbf{1}[d = 2] + \varepsilon_{0} & \text{if } a = 0 \\
   -\theta_4 + \varepsilon_{1} & \text{if } a = 1
   \end{cases}

where :math:`\varepsilon_a` are i.i.d. Type I Extreme Value shocks. The parameter :math:`\theta_1` is the per-year operating cost that captures the rising maintenance burden of aging vehicles. The parameters :math:`\theta_2` and :math:`\theta_3` are the cost penalties for minor and major APK defects respectively. The parameter :math:`\theta_4` is the net replacement cost of scrapping the current vehicle and purchasing a new one.

The state transitions have two components. Age increments deterministically by one year when the owner keeps the car and resets to zero upon scrappage. Defect severity transitions stochastically with probabilities that depend on age. Older vehicles are more likely to develop defects in the next period. Formally, the transition probability from state :math:`(a, d)` under the keep action is

.. math::

   P\big((\text{age}+1, d') \mid (\text{age}, d), a=0\big) = \pi_{d \to d'}(\text{age})

where :math:`\pi_{d \to d'}(\text{age})` is the age-dependent defect transition matrix. Under the scrap action the state resets to :math:`(0, 0)` with probability one.

The owner solves the Bellman equation

.. math::

   V(s) = \sigma \log \sum_{a \in \{0,1\}} \exp\!\Big(\frac{u(s,a;\theta) + \beta \sum_{s'} P(s' \mid s,a) V(s')}{\sigma}\Big)

where :math:`\beta = 0.95` is the annual discount factor and :math:`\sigma = 1` is the logit scale parameter. The implied choice probability is

.. math::

   \Pr(a \mid s; \theta) = \frac{\exp\!\big(u(s,a;\theta) + \beta \sum_{s'} P(s' \mid s,a) V(s')\big)}{\sum_{a'} \exp\!\big(u(s,a';\theta) + \beta \sum_{s'} P(s' \mid s,a') V(s')\big)}

The state space has :math:`25 \times 3 = 75` states and the model has 4 structural parameters. Age cost and replacement cost are identified from the age profile of scrappage rates. Defect costs are identified from the differential scrappage rates across defect levels conditional on age.

Estimation
----------

Three estimators recover the structural parameters. NFXP maximizes the log-likelihood directly by solving the Bellman equation in an inner loop at each parameter evaluation. CCP inverts the Hotz-Miller mapping and iterates via nested pseudo-likelihood. GLADIUS trains neural Q-networks and EV-networks with a Bellman consistency penalty, then projects the implied rewards onto the feature matrix to recover structural parameters.

.. code-block:: python

   from econirl.environments.rdw_scrappage import RDWScrapageEnvironment
   from econirl.datasets import load_rdw_scrappage
   from econirl.estimation.nfxp import NFXPEstimator
   from econirl.estimation.ccp import CCPEstimator
   from econirl.estimation.gladius import GLADIUSEstimator, GLADIUSConfig
   from econirl.preferences.linear import LinearUtility

   env = RDWScrapageEnvironment(discount_factor=0.95)
   panel = load_rdw_scrappage(data_dir="src/econirl/datasets/", as_panel=True)
   utility = LinearUtility.from_environment(env)

   # NFXP: nested fixed point with BHHH optimizer
   nfxp = NFXPEstimator(optimizer="BHHH", inner_solver="policy", compute_hessian=True)
   result_nfxp = nfxp.estimate(panel, utility, env.problem_spec, transitions)

   # CCP: Hotz-Miller with NPL iterations
   ccp = CCPEstimator(num_policy_iterations=20, compute_hessian=True)
   result_ccp = ccp.estimate(panel, utility, env.problem_spec, transitions)

   # GLADIUS: neural Q-learning with Bellman penalty
   gladius = GLADIUSEstimator(config=GLADIUSConfig(max_epochs=300, compute_se=True))
   result_gladius = gladius.estimate(panel, utility, env.problem_spec, transitions)

NFXP recovers age_cost of 0.115 with a standard error of 0.007 and a t-statistic of 17.2. The replacement cost estimate is 8.54 with a standard error of 0.25 and a t-statistic of 34.9. Major defect cost is 0.37 with a t-statistic of 2.6, significant at the 1 percent level. Minor defect cost is not statistically significant, indicating that minor APK defects like worn tires or dim bulbs do not materially influence scrappage decisions. CCP and GLADIUS produce estimates in the same range.

Counterfactual analysis
-----------------------

The estimated model supports four types of counterfactual analysis. Each counterfactual solves a new Bellman equation under the modified primitives and compares the implied policy and welfare to the baseline.

**Scrappage subsidy.** A government policy that subsidizes 30 percent of the replacement cost reduces the effective :math:`\theta_4` from 8.54 to 5.98. The subsidy increases scrappage rates at every age, with the largest effect on old vehicles with major defects where the owner was previously on the margin.

**Defect deterioration.** If road conditions worsen (for example due to climate-driven infrastructure damage), the defect transition probabilities shift so that vehicles age into worse defect states faster. Doubling the defect-age sensitivity parameter from 0.02 to 0.04 raises the average defect severity in the fleet and increases scrappage at every age.

**Elasticity.** The elasticity of the scrappage rate to the replacement cost measures how sensitive the fleet turnover rate is to the price of new vehicles. Varying the replacement cost by plus or minus 10 to 50 percent traces out the policy response surface.

**Welfare decomposition.** The total welfare change from a policy intervention decomposes into a direct effect (value change holding the stationary distribution fixed) and a distribution effect (change in the stationary distribution of states). The subsidy improves welfare both directly by making replacement cheaper for agents at every state and indirectly by shifting the fleet toward younger vehicles with fewer defects.

.. code-block:: python

   from econirl.simulation.counterfactual import (
       counterfactual_policy,
       counterfactual_transitions,
       elasticity_analysis,
       compute_welfare_effect,
       simulate_counterfactual,
   )

   # Scrappage subsidy: reduce replacement cost by 30%
   new_params = result_nfxp.parameters.at[3].mul(0.7)
   cf = counterfactual_policy(result_nfxp, new_params, utility, problem, transitions)
   welfare = compute_welfare_effect(cf, transitions)

   # Defect deterioration: double the age-sensitivity
   worse_env = RDWScrapageEnvironment(defect_age_sensitivity=0.04)
   cf_worse = counterfactual_transitions(
       result_nfxp, worse_env.transition_matrices, utility, problem, transitions
   )

   # Elasticity of scrappage to replacement cost
   ea = elasticity_analysis(
       result_nfxp, utility, problem, transitions,
       parameter_name="replacement_cost",
       pct_changes=[-0.50, -0.30, -0.10, 0.10, 0.30, 0.50],
   )

Using real RDW data
-------------------

The RDW makes vehicle registration and inspection data freely available under a CC-0 license at opendata.rdw.nl. The download script queries the Socrata SODA API for a specific vehicle cohort, downloads defect records, and builds the car-year panel. By default it downloads Volkswagen Golf models registered between 2005 and 2015.

.. code-block:: bash

   # Download default cohort (VW Golf 2005-2015, ~170K vehicles)
   python scripts/download_rdw.py

   # Download a different cohort
   python scripts/download_rdw.py --brand TOYOTA --model COROLLA

   # Run the full showcase (estimation + counterfactuals)
   python examples/rdw-scrappage/rdw_showcase.py --data-dir src/econirl/datasets/

Running the example
-------------------

.. code-block:: bash

   # Full showcase with synthetic fallback (no download needed)
   python examples/rdw-scrappage/rdw_showcase.py

   # Quick test with 500 vehicles
   python examples/rdw-scrappage/rdw_showcase.py --max-vehicles 500

   # NFXP only (faster)
   python examples/rdw-scrappage/rdw_nfxp.py

   # With real RDW data
   python examples/rdw-scrappage/rdw_showcase.py --data-dir src/econirl/datasets/ --max-vehicles 5000

Reference
---------

RDW Open Data. Rijksdienst voor het Wegverkeer. https://opendata.rdw.nl

El Boubsi, M. (2023). Predicting Vehicle Inspection Outcomes Using Machine Learning. MSc Thesis, Delft University of Technology.

Rust, J. (1987). Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher. Econometrica, 55(5), 999-1033.

Kang, M., Yoganarasimhan, H., and Jain, V. (2025). Efficient Estimation of Random Coefficients Demand Models using Machine Learning. Working Paper.
