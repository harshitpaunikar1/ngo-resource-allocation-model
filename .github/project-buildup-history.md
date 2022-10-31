# Project Buildup History: NGO Resource Allocation Model

- Repository: `ngo-resource-allocation-model`
- Category: `data_science`
- Subtype: `optimization`
- Source: `project_buildup_2021_2025_daily_plan_extra.csv`
## 2022-09-12 - Day 2: Problem structuring

- Task summary: Got back to the NGO Resource Allocation model today. The core challenge is allocating limited program budget across geography and intervention type to maximize outcome impact. Spent today formalizing the problem structure — what the decision variables are, what the constraints look like (budget caps, geographic coverage minimums, intervention type balance requirements), and how to measure the objective. Also gathered some reference allocation schemes from published NGO reports to use as sanity checks against the model output.
- Deliverable: Problem structure formalized. Decision variables and constraints documented.
## 2022-09-12 - Day 2: Problem structuring

- Task summary: Added a parameter sensitivity section outline — wanted to make sure the final model would have a clear way to test how sensitive the allocation recommendations are to changes in the impact estimates.
- Deliverable: Sensitivity analysis structure planned. Will implement after base model is working.
## 2022-09-12 - Day 2: Problem structuring

- Task summary: Quick data check: the impact estimate table had a few missing region-intervention combinations. Decided to impute those with the regional average rather than dropping the region entirely.
- Deliverable: Missing impact estimates imputed with regional averages.
## 2022-09-19 - Day 3: Linear programming baseline

- Task summary: Implemented a linear programming baseline for the NGO allocation problem today using scipy's linprog. The LP relaxation ignores the integer constraints on some allocation variables but gives a useful lower bound and is fast to solve. The unconstrained optimal solution was not surprising — it concentrated resources heavily in the highest-impact interventions. Added the geographic coverage constraint and saw how that changed the allocation spread. Documented both scenarios.
- Deliverable: LP baseline with and without geographic constraints implemented and documented.
## 2022-10-31 - Day 4: Sensitivity analysis

- Task summary: Ran the sensitivity analysis for the NGO allocation model today. Varied the impact estimates by plus and minus 20 percent and observed how the optimal allocation shifted. Some interventions were robust — they appeared in the top allocations across all scenarios. Others were sensitive — they only appeared when their impact estimate was at the optimistic end. Plotted a tornado diagram to show the range of outcome variation per parameter. That diagram became the most useful output for the stakeholder framing.
- Deliverable: Sensitivity analysis complete. Tornado diagram added. Robust vs sensitive interventions identified.
