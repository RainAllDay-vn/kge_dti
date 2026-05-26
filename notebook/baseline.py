# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "scikit-learn",
# ]
# ///

import marimo

__generated_with = "0.23.7"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Drug-Target Interaction (DTI) Prediction
    ### Dataset Exploration & Baseline Establishment

    Predicting interactions between drugs and target proteins is a crucial step in modern *in silico* drug discovery. This notebook serves as an interactive environment to inspect data quality and establish a model baseline.

    ## Core Notebook Goals:

    - **Explore the Dataset:** Investigate DTI interaction networks, Morgan drug structure fingerprints, protein sequence CTD descriptors, and Knowledge Graph relation triples within the `data/yamanishi_08` directory.
    - **Establish a Baseline:** Construct and validate a robust Random Forest classifier baseline to predict drug-target interactions across cross-validation splits.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    # Drug-Target Interaction (DTI) Prediction
    ### Dataset Exploration & Baseline Establishment

    Predicting interactions between drugs and target proteins is a crucial step in modern *in silico* drug discovery. This notebook serves as an interactive environment to inspect data quality and establish a model baseline.

    ## Core Notebook Goals:

    - **Explore the Dataset:** Investigate DTI interaction networks, Morgan drug structure fingerprints, protein sequence CTD descriptors, and Knowledge Graph relation triples within the `data/yamanishi_08` directory.
    - **Establish a Baseline:** Construct and validate a robust Random Forest classifier baseline to predict drug-target interactions across cross-validation splits.
    """)
    return


@app.cell
def _():
    # Set global variables
    dataset_root = Path("./data/yamanishi_08")
    return (dataset_root,)


@app.cell
def _():
    mo.md(r"""
    ## 1. Visualizing Dataset Shapes and Structures

    The `yamanishi_08` dataset includes: positive Drug-Target Interactions (DTIs), drug structures and precomputed chemical fingerprints, protein sequences and CTD sequence descriptors, and Knowledge Graph (KG) relation triples.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### 1.1 Drug-Target Interactions (`dt_all_08.txt`)

    This file defines the gold-standard bipartite drug-target interaction (DTI) network from the Yamanishi 08 benchmark dataset.

    - **Data Format**: Tab/whitespace-separated values with 3 columns:
      1. `drug_id`: Bipartite node representing the drug compound (e.g., `D00002`, KEGG Drug ID).
      2. `relation`: Always set to `DRUG_TARGET`, defining the positive interaction link.
      3. `target_id`: Bipartite node representing the target protein (e.g., `hsa:10`, KEGG Gene ID).
    - **Machine Learning Context**: These pairs serve as the positive class ground-truth labels for DTI binary classification or link prediction tasks. All unobserved edges are generally treated as negative samples during baseline model training.
    """)
    return


@app.cell
def _(dataset_root):
    dti_df = pd.read_csv(
        dataset_root / "dt_all_08.txt",
        sep=r"\s+",
        header=None,
        names=["drug_id", "relation", "target_id"]
    )
    return (dti_df,)


@app.cell
def _(dti_df):
    def _display_stats():
        num_interactions = len(dti_df)
        num_drugs = dti_df["drug_id"].nunique()
        num_targets = dti_df["target_id"].nunique()

        matrix_size = num_drugs * num_targets
        density = num_interactions / matrix_size
        sparsity = 1.0 - density

        avg_targets_per_drug = num_interactions / num_drugs
        avg_drugs_per_target = num_interactions / num_targets

        stats_df = pd.DataFrame({
            "Metric": [
                "Total Interactions",
                "Unique Drugs",
                "Unique Target Proteins",
                "Matrix Density",
                "Matrix Sparsity",
                "Average Targets per Drug",
                "Average Drugs per Target Protein"
            ],
            "Value": [
                f"{num_interactions:,}",
                f"{num_drugs:,}",
                f"{num_targets:,}",
                f"{density:.4%}",
                f"{sparsity:.2%}",
                round(avg_targets_per_drug, 2),
                round(avg_drugs_per_target, 2)
            ]
        })

        return mo.vstack([
            mo.md("### DTI Dataset Summary Statistics"),
            stats_df
        ])

    _display_stats()
    return


@app.cell
def _(dti_df):
    def _display_degree_stats():
        drug_degrees = dti_df.groupby("drug_id").size()
        target_degrees = dti_df.groupby("target_id").size()

        # Styling matplotlib for dark/sleek theme
        plt.rcParams["figure.facecolor"] = "none"
        plt.rcParams["axes.facecolor"] = "none"
        plt.rcParams["text.color"] = "#E2E8F0"
        plt.rcParams["axes.labelcolor"] = "#94A3B8"
        plt.rcParams["xtick.color"] = "#94A3B8"
        plt.rcParams["ytick.color"] = "#94A3B8"
        plt.rcParams["grid.color"] = "#334155"
        plt.rcParams["axes.edgecolor"] = "#475569"

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Drug Degree Distribution
        axes[0].hist(drug_degrees, bins=30, color="#8b5cf6", alpha=0.8, edgecolor="#1e1b4b", rwidth=0.85)
        axes[0].set_title("Drug Degree Distribution", fontsize=12, fontweight="bold", pad=15, color="#F1F5F9")
        axes[0].set_xlabel("Number of Associated Targets", fontsize=10)
        axes[0].set_ylabel("Count of Drugs", fontsize=10)
        axes[0].grid(True, linestyle="--", alpha=0.3)

        # Target Degree Distribution
        axes[1].hist(target_degrees, bins=30, color="#d946ef", alpha=0.8, edgecolor="#311042", rwidth=0.85)
        axes[1].set_title("Protein Target Degree Distribution", fontsize=12, fontweight="bold", pad=15, color="#F1F5F9")
        axes[1].set_xlabel("Number of Associated Drugs", fontsize=10)
        axes[1].set_ylabel("Count of Targets", fontsize=10)
        axes[1].grid(True, linestyle="--", alpha=0.3)

        plt.tight_layout()
        plot_ui = mo.as_html(fig)
        plt.close(fig)

        return mo.vstack([
            mo.md(r"""
            ### 1.1.1 Degree Distributions & DTI Network Sparsity

            To understand the connectivity landscape, we analyze the degree distribution of the bipartite DTI network. The degree of a drug represents the number of target proteins it is known to interact with, while the degree of a protein represents the number of active drugs targeting it.
            """),
            plot_ui,
            mo.md(r"""
            #### **Key Insights & Structural Observations:**

            1. **Low-Degree Node Domination (High Sparsity)**:
                - **Drugs**: Nearly **74.6% of all drugs** (590 out of 791) have a degree of 5 or fewer, meaning the vast majority of drugs are associated with very few targets.
                - **Targets**: More than **54.6% of all target proteins** (540 out of 989) interact with 2 or fewer drugs.
            2. **Power-Law/Scale-Free Characteristics**:
                - Both distributions exhibit a heavy-tailed decay. A tiny group of highly connected "hub" nodes (e.g., a few drugs targeting over 100 proteins) hold the network together, while most nodes have very sparse connections.
            3. **Interaction Matrix Density ($0.655\%$)**:
                - Out of $782,301$ possible bipartite edges ($791 \text{ drugs} \times 989 \text{ targets}$), only **5,128** are active. This extreme sparsity ($99.345\%$ empty space) presents a classic sparse matrix completion challenge for downstream DTI machine learning models.
            """)
        ])

    _display_degree_stats()
    return


@app.cell
def _(dti_df):
    def _display_hub_view():
        top_drugs = dti_df.groupby("drug_id").size().reset_index(name="degree").sort_values(by="degree", ascending=False).head(10)
        top_targets = dti_df.groupby("target_id").size().reset_index(name="degree").sort_values(by="degree", ascending=False).head(10)

        top_drugs_table = mo.ui.table(top_drugs, label="Top 10 Most Interactive Drugs")
        top_targets_table = mo.ui.table(top_targets, label="Top 10 Most Interactive Targets")

        hub_view = mo.hstack([top_drugs_table, top_targets_table], justify="space-around")

        return mo.vstack([
            mo.md(r"""
            ### DTI Network Hub Identification
            Below are the most interactive entities (hubs) in the DTI network:
            """),
            hub_view
        ])

    _display_hub_view()
    return


@app.cell
def _():
    mo.md(r"""
    ### 1.2 Drug Structure Mapping (`791drug_struc.csv`):
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### 1.3 Protein Sequence Mapping (`989proseq.csv`):
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### 1.4 Combined Knowledge Graph relation triples:
    """)
    return


if __name__ == "__main__":
    app.run()
