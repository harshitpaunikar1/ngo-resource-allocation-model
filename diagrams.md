# NGO Data-Driven Resource Allocation Diagrams

Generated on 2026-04-26T04:17:39Z from repository evidence.

## Architecture Overview

```mermaid
flowchart LR
    A[Repository Inputs] --> B[Preparation and Validation]
    B --> C[ML Case Study Core Logic]
    C --> D[Output Surface]
    D --> E[Insights or Actions]
```

## Workflow Sequence

```mermaid
flowchart TD
    S1["Importing libraries"]
    S2["Importing  Country-data.csv file"]
    S1 --> S2
    S3["Data exploration"]
    S2 --> S3
    S4["checking correlation between the features"]
    S3 --> S4
    S5["Plotting the co-relation heatmap"]
    S4 --> S5
```
