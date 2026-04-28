"""
Allocation runner for NGO resource optimization.
Loads field data, runs the optimizer, and generates impact reports.
"""
import json
import logging
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from optimizer import (
    AllocationDecision, GreedyAllocator, ImpactScorer,
    LPAllocator, NeedCluster, Resource
)

logger = logging.getLogger(__name__)


class FieldDataLoader:
    """Loads resources and clusters from dictionaries or DataFrames."""

    def load_resources(self, data: List[Dict]) -> List[Resource]:
        resources = []
        for r in data:
            resources.append(Resource(
                resource_id=str(r["resource_id"]),
                name=str(r["name"]),
                resource_type=str(r["resource_type"]),
                quantity=float(r["quantity"]),
                cost_per_unit=float(r["cost_per_unit"]),
                capacity=float(r.get("capacity", 1)),
                region_constraints=r.get("region_constraints", []),
            ))
        return resources

    def load_clusters(self, data: List[Dict]) -> List[NeedCluster]:
        clusters = []
        for c in data:
            clusters.append(NeedCluster(
                cluster_id=str(c["cluster_id"]),
                region=str(c["region"]),
                beneficiaries=int(c["beneficiaries"]),
                urgency=float(c["urgency"]),
                required_resources=c.get("required_resources", {}),
                days_until_critical=int(c.get("days_until_critical", 30)),
            ))
        return clusters

    def from_dataframes(self, resources_df: pd.DataFrame,
                         clusters_df: pd.DataFrame) -> tuple:
        resources = self.load_resources(resources_df.to_dict("records"))
        clusters = self.load_clusters(clusters_df.to_dict("records"))
        return resources, clusters


class AllocationReport:
    """Generates structured impact and cost reports from allocation decisions."""

    def __init__(self, decisions: List[AllocationDecision],
                 clusters: List[NeedCluster], resources: List[Resource],
                 total_budget: float):
        self.decisions = decisions
        self.clusters = {c.cluster_id: c for c in clusters}
        self.resources = {r.resource_id: r for r in resources}
        self.total_budget = total_budget

    def cluster_coverage_report(self) -> pd.DataFrame:
        df = pd.DataFrame([d.__dict__ for d in self.decisions])
        if df.empty:
            return pd.DataFrame()
        grp = df.groupby("cluster_id").agg(
            total_units=("units_allocated", "sum"),
            total_cost=("cost", "sum"),
            avg_impact=("impact_score", "mean"),
            num_resources=("resource_id", "nunique"),
        ).reset_index()
        grp["cluster_beneficiaries"] = grp["cluster_id"].map(
            {cid: c.beneficiaries for cid, c in self.clusters.items()}
        )
        grp["cost_per_beneficiary"] = (
            grp["total_cost"] / grp["cluster_beneficiaries"].replace(0, np.nan)
        ).round(2)
        return grp.sort_values("avg_impact", ascending=False).reset_index(drop=True)

    def resource_utilization_report(self) -> pd.DataFrame:
        df = pd.DataFrame([d.__dict__ for d in self.decisions])
        if df.empty:
            return pd.DataFrame()
        grp = df.groupby("resource_id").agg(
            units_deployed=("units_allocated", "sum"),
            total_cost=("cost", "sum"),
            clusters_served=("cluster_id", "nunique"),
        ).reset_index()
        grp["total_supply"] = grp["resource_id"].map(
            {r.resource_id: r.quantity for r in self.resources.values()}
        )
        grp["utilization_pct"] = (
            grp["units_deployed"] / grp["total_supply"].replace(0, np.nan) * 100
        ).round(1)
        return grp.sort_values("utilization_pct", ascending=False).reset_index(drop=True)

    def executive_summary(self) -> Dict:
        total_cost = sum(d.cost for d in self.decisions)
        total_impact = sum(d.impact_score for d in self.decisions)
        covered = set(d.cluster_id for d in self.decisions)
        uncovered = set(self.clusters.keys()) - covered
        beneficiaries_reached = sum(
            self.clusters[cid].beneficiaries for cid in covered
        )
        total_beneficiaries = sum(c.beneficiaries for c in self.clusters.values())
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "total_budget": self.total_budget,
            "total_spent": round(total_cost, 2),
            "budget_remaining": round(self.total_budget - total_cost, 2),
            "clusters_covered": len(covered),
            "clusters_uncovered": len(uncovered),
            "beneficiaries_reached": beneficiaries_reached,
            "total_beneficiaries": total_beneficiaries,
            "coverage_pct": round(beneficiaries_reached / max(total_beneficiaries, 1) * 100, 1),
            "total_impact_score": round(total_impact, 4),
            "cost_per_impact_unit": round(total_cost / max(total_impact, 0.001), 2),
        }


class AllocationRunner:
    """
    Orchestrates field data loading, optimization, and report generation.
    Compares greedy and LP allocations.
    """

    def __init__(self, total_budget: float, method: str = "lp"):
        self.total_budget = total_budget
        self.method = method.lower()
        self.scorer = ImpactScorer()
        self.loader = FieldDataLoader()

    def run(self, resources: List[Resource], clusters: List[NeedCluster]) -> AllocationReport:
        if self.method == "lp":
            allocator = LPAllocator(resources, clusters, self.total_budget, self.scorer)
        else:
            allocator = GreedyAllocator(resources, clusters, self.total_budget, self.scorer)
        decisions = allocator.allocate()
        return AllocationReport(decisions, clusters, resources, self.total_budget)

    def compare_methods(self, resources: List[Resource],
                         clusters: List[NeedCluster]) -> pd.DataFrame:
        results = []
        for method in ["greedy", "lp"]:
            runner = AllocationRunner(self.total_budget, method=method)
            report = runner.run(resources, clusters)
            summary = report.executive_summary()
            summary["method"] = method
            results.append(summary)
        return pd.DataFrame(results)[
            ["method", "total_spent", "coverage_pct", "beneficiaries_reached", "total_impact_score"]
        ]


if __name__ == "__main__":
    np.random.seed(42)
    resources_data = [
        {"resource_id": "R001", "name": "Medical Team", "resource_type": "medical_staff",
         "quantity": 30, "cost_per_unit": 300, "capacity": 20},
        {"resource_id": "R002", "name": "Food Kits", "resource_type": "supply_kit",
         "quantity": 500, "cost_per_unit": 40, "capacity": 50},
        {"resource_id": "R003", "name": "Relief Van", "resource_type": "vehicle",
         "quantity": 8, "cost_per_unit": 600, "capacity": 100, "region_constraints": ["North", "East"]},
        {"resource_id": "R004", "name": "Water Units", "resource_type": "water_unit",
         "quantity": 80, "cost_per_unit": 120, "capacity": 200},
    ]
    clusters_data = [
        {"cluster_id": "C01", "region": "North", "beneficiaries": 6000, "urgency": 0.95,
         "required_resources": {"medical_staff": 8, "supply_kit": 60}, "days_until_critical": 3},
        {"cluster_id": "C02", "region": "South", "beneficiaries": 4000, "urgency": 0.70,
         "required_resources": {"supply_kit": 40, "water_unit": 20}, "days_until_critical": 14},
        {"cluster_id": "C03", "region": "East", "beneficiaries": 7000, "urgency": 0.85,
         "required_resources": {"vehicle": 2, "medical_staff": 10}, "days_until_critical": 6},
        {"cluster_id": "C04", "region": "West", "beneficiaries": 2500, "urgency": 0.55,
         "required_resources": {"supply_kit": 25, "water_unit": 30}, "days_until_critical": 25},
    ]

    loader = FieldDataLoader()
    resources = loader.load_resources(resources_data)
    clusters = loader.load_clusters(clusters_data)

    runner = AllocationRunner(total_budget=60000, method="lp")
    report = runner.run(resources, clusters)

    print("Executive Summary:")
    print(json.dumps(report.executive_summary(), indent=2))

    print("\nCluster Coverage:")
    print(report.cluster_coverage_report().to_string(index=False))

    print("\nResource Utilization:")
    print(report.resource_utilization_report().to_string(index=False))

    print("\nMethod Comparison (Greedy vs LP):")
    print(runner.compare_methods(resources, clusters).to_string(index=False))
