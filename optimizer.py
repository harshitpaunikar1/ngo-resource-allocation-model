"""
Resource allocation optimizer for NGO field operations.
Solves a multi-constraint assignment problem to maximize impact per rupee/dollar.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from scipy.optimize import linprog, milp, LinearConstraint, Bounds
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class Resource:
    resource_id: str
    name: str
    resource_type: str      # volunteer, vehicle, supply_kit, medical_staff
    quantity: float
    cost_per_unit: float
    capacity: float         # units of work per day
    region_constraints: List[str] = field(default_factory=list)  # allowed regions


@dataclass
class NeedCluster:
    cluster_id: str
    region: str
    beneficiaries: int
    urgency: float          # 0-1, higher = more urgent
    required_resources: Dict[str, float]  # resource_type -> min units needed
    days_until_critical: int = 30


@dataclass
class AllocationDecision:
    cluster_id: str
    resource_id: str
    units_allocated: float
    cost: float
    impact_score: float


class ImpactScorer:
    """Scores the impact of allocating a resource to a cluster."""

    def __init__(self, urgency_weight: float = 0.5,
                 beneficiary_weight: float = 0.3,
                 efficiency_weight: float = 0.2):
        total = urgency_weight + beneficiary_weight + efficiency_weight
        self.urgency_w = urgency_weight / total
        self.beneficiary_w = beneficiary_weight / total
        self.efficiency_w = efficiency_weight / total

    def score(self, cluster: NeedCluster, resource: Resource,
              units: float, total_beneficiaries: float) -> float:
        urgency_score = cluster.urgency
        beneficiary_score = cluster.beneficiaries / max(total_beneficiaries, 1)
        efficiency_score = min(1.0, (units * resource.capacity) / max(cluster.beneficiaries, 1))
        return (self.urgency_w * urgency_score
                + self.beneficiary_w * beneficiary_score
                + self.efficiency_w * efficiency_score)


class GreedyAllocator:
    """
    Greedy allocation: iteratively assign resources to highest-priority clusters
    until budget or supply is exhausted.
    """

    def __init__(self, resources: List[Resource], clusters: List[NeedCluster],
                 total_budget: float, scorer: Optional[ImpactScorer] = None):
        self.resources = resources
        self.clusters = clusters
        self.total_budget = total_budget
        self.scorer = scorer or ImpactScorer()
        self._decisions: List[AllocationDecision] = []

    def allocate(self) -> List[AllocationDecision]:
        remaining_budget = self.total_budget
        remaining_supply = {r.resource_id: r.quantity for r in self.resources}
        unmet_clusters = list(self.clusters)
        total_beneficiaries = sum(c.beneficiaries for c in self.clusters)

        priority_queue = sorted(
            unmet_clusters,
            key=lambda c: c.urgency * c.beneficiaries / max(c.days_until_critical, 1),
            reverse=True,
        )

        for cluster in priority_queue:
            for resource in self.resources:
                if remaining_supply[resource.resource_id] <= 0:
                    continue
                if resource.region_constraints and cluster.region not in resource.region_constraints:
                    continue
                required = cluster.required_resources.get(resource.resource_type, 0)
                if required <= 0:
                    continue
                affordable = remaining_budget / resource.cost_per_unit
                available = min(remaining_supply[resource.resource_id], affordable)
                units = min(required, available)
                if units <= 0:
                    continue
                cost = units * resource.cost_per_unit
                impact = self.scorer.score(cluster, resource, units, total_beneficiaries)
                self._decisions.append(AllocationDecision(
                    cluster_id=cluster.cluster_id,
                    resource_id=resource.resource_id,
                    units_allocated=round(units, 2),
                    cost=round(cost, 2),
                    impact_score=round(impact, 4),
                ))
                remaining_supply[resource.resource_id] -= units
                remaining_budget -= cost

        return self._decisions

    def summary(self) -> Dict:
        if not self._decisions:
            return {}
        total_cost = sum(d.cost for d in self._decisions)
        total_impact = sum(d.impact_score for d in self._decisions)
        covered = len(set(d.cluster_id for d in self._decisions))
        return {
            "total_decisions": len(self._decisions),
            "clusters_covered": covered,
            "total_clusters": len(self.clusters),
            "total_cost": round(total_cost, 2),
            "avg_impact_score": round(total_impact / max(len(self._decisions), 1), 4),
            "budget_utilization_pct": round(total_cost / self.total_budget * 100, 1),
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([d.__dict__ for d in self._decisions])


class LPAllocator:
    """
    LP-based allocation for maximizing weighted impact subject to budget and supply constraints.
    Falls back to greedy if scipy is unavailable.
    """

    def __init__(self, resources: List[Resource], clusters: List[NeedCluster],
                 total_budget: float, scorer: Optional[ImpactScorer] = None):
        self.resources = resources
        self.clusters = clusters
        self.total_budget = total_budget
        self.scorer = scorer or ImpactScorer()
        self._decisions: List[AllocationDecision] = []

    def allocate(self) -> List[AllocationDecision]:
        if not SCIPY_AVAILABLE:
            logger.warning("scipy unavailable, falling back to greedy.")
            fallback = GreedyAllocator(self.resources, self.clusters,
                                        self.total_budget, self.scorer)
            self._decisions = fallback.allocate()
            return self._decisions

        total_beneficiaries = sum(c.beneficiaries for c in self.clusters)
        pairs = [
            (c, r)
            for c in self.clusters
            for r in self.resources
            if c.required_resources.get(r.resource_type, 0) > 0
            and (not r.region_constraints or c.region in r.region_constraints)
        ]
        n = len(pairs)
        if n == 0:
            return []

        c_obj = -np.array([
            self.scorer.score(cluster, resource, 1.0, total_beneficiaries)
            for cluster, resource in pairs
        ])

        A_budget = np.array([resource.cost_per_unit for _, resource in pairs]).reshape(1, n)
        b_budget = np.array([self.total_budget])

        bounds = []
        for cluster, resource in pairs:
            max_needed = cluster.required_resources.get(resource.resource_type, 0)
            max_available = resource.quantity
            bounds.append((0, min(max_needed, max_available)))

        result = linprog(c_obj, A_ub=A_budget, b_ub=b_budget, bounds=bounds, method="highs")
        if not result.success:
            logger.warning("LP did not converge: %s. Falling back to greedy.", result.message)
            fallback = GreedyAllocator(self.resources, self.clusters,
                                        self.total_budget, self.scorer)
            self._decisions = fallback.allocate()
            return self._decisions

        for (cluster, resource), units in zip(pairs, result.x):
            if units < 0.01:
                continue
            impact = self.scorer.score(cluster, resource, units, total_beneficiaries)
            self._decisions.append(AllocationDecision(
                cluster_id=cluster.cluster_id,
                resource_id=resource.resource_id,
                units_allocated=round(float(units), 2),
                cost=round(float(units) * resource.cost_per_unit, 2),
                impact_score=round(impact, 4),
            ))
        return self._decisions

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([d.__dict__ for d in self._decisions])


if __name__ == "__main__":
    np.random.seed(42)
    resources = [
        Resource("R001", "Medical Volunteers", "medical_staff", 50, 200, 20),
        Resource("R002", "Supply Kits", "supply_kit", 200, 80, 50),
        Resource("R003", "Relief Vehicles", "vehicle", 10, 500, 100, ["North", "East"]),
        Resource("R004", "Clean Water Units", "water_unit", 100, 150, 200),
    ]

    clusters = [
        NeedCluster("C01", "North", 5000, 0.9, {"medical_staff": 10, "supply_kit": 50}, 5),
        NeedCluster("C02", "South", 3000, 0.7, {"supply_kit": 30, "water_unit": 20}, 15),
        NeedCluster("C03", "East", 8000, 0.85, {"vehicle": 3, "medical_staff": 15}, 7),
        NeedCluster("C04", "West", 2000, 0.5, {"supply_kit": 20, "water_unit": 30}, 30),
        NeedCluster("C05", "North", 4000, 0.75, {"medical_staff": 8, "supply_kit": 40}, 10),
    ]

    scorer = ImpactScorer(urgency_weight=0.5, beneficiary_weight=0.3, efficiency_weight=0.2)
    allocator = LPAllocator(resources=resources, clusters=clusters,
                             total_budget=50000, scorer=scorer)
    decisions = allocator.allocate()
    df = allocator.to_dataframe()
    print("Allocation decisions:")
    print(df.sort_values("impact_score", ascending=False).to_string(index=False))

    greedy = GreedyAllocator(resources=resources, clusters=clusters,
                              total_budget=50000, scorer=scorer)
    greedy.allocate()
    print("\nGreedy summary:", greedy.summary())
