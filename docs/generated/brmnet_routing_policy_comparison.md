# BRM-Net routing policy comparison

Houston2013 routing profiles use the formal 65/80/100 compact profile bank across three seeds. Mean OA and MACs are averaged over the evaluated modality/degradation states.

| Policy | Type | Mean OA | Mean MACs | Regret | Saving vs 100% | Routing acc. | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Static 100% | fixed profile | 0.7483 +/- 0.0178 | 1.0000 +/- 0.0000 | n/a | 0.0000 | n/a | Highest fixed-profile baseline and reference cost. |
| Static 80% | fixed profile | 0.7399 +/- 0.0158 | 0.8036 +/- 0.0001 | n/a | 0.1964 | n/a | Strong low-cost baseline; hard to beat with learned routing. |
| Pareto oracle delta=0.01 | oracle label | 0.7537 +/- 0.0164 | 0.8810 +/- 0.0403 | 0.0010 | 0.1190 | 1.0000 | Best current routing target; not deployable by itself. |
| State-level LOO router | learned state-level | 0.7464 | 0.8929 | 0.0082 | 0.1071 | 0.6667 | Generalizes partly, but uses only 11 state samples per seed. |
| Validation-state-derived router | learned seed/state-level | 0.7459 | 0.8622 | 0.0243 | 0.0799 | 0.3333 | Similar OA at lower MACs, but label instability hurts routing accuracy. |

## Current conclusion

The routing evidence should be written as a staged result rather than a solved module. Pareto labels define a strong accuracy-efficiency target, but learned routing is still limited by supervision quality. The validation-state-derived experiment expands supervision from 11 state samples to 33 seed/state samples and confirms that the next bottleneck is label stability: most degradation states receive different Pareto budgets across seeds.

The most defensible next step is a true patch-level routing dataset with stored quality-probe outputs, prediction confidence, modality state, and profile-bank outcomes.
