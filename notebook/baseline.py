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
    ### 1.2 Drug Structure Mapping & Chemical Fingerprints

    To represent drug compounds mathematically, we map each unique drug identifier to its chemical structure and precomputed molecular fingerprints:

    1. **`791drug_struc.csv`**: Contains canonical SMILES (Simplified Molecular Input Line Entry System) strings, representing 2D chemical structure graphs.
    2. **`morganfp.txt`**: Contains 1024-dimensional topological circular Morgan fingerprints (similar to ECFP4, radius 2), representing local atomic neighborhoods.
    """)
    return


@app.cell
def _(dataset_root):
    drug_struc_df = pd.read_csv(dataset_root / "791drug_struc.csv")
    drug_fps = np.loadtxt(dataset_root / "morganfp.txt", delimiter=",")
    return drug_fps, drug_struc_df


@app.cell
def _(drug_fps, drug_struc_df):
    def _display_drug_stats():
        # Compute SMILES lengths
        drug_struc_df_with_len = drug_struc_df.copy()
        drug_struc_df_with_len["smiles_length"] = drug_struc_df_with_len["smiles"].apply(len)

        num_drugs = len(drug_struc_df_with_len)
        avg_smiles_len = drug_struc_df_with_len["smiles_length"].mean()
        max_smiles_len = drug_struc_df_with_len["smiles_length"].max()
        min_smiles_len = drug_struc_df_with_len["smiles_length"].min()

        # Morgan Fingerprint density
        active_bits_per_drug = np.sum(drug_fps == 1.0, axis=1)
        avg_active_bits = np.mean(active_bits_per_drug)
        global_sparsity = 1.0 - (np.sum(drug_fps == 1.0) / drug_fps.size)

        stats_df = pd.DataFrame({
            "Metric": [
                "Total Compounds",
                "Average SMILES Length",
                "Max SMILES Length",
                "Min SMILES Length",
                "Morgan FP Dimensions",
                "Average Active Bits (per Drug)",
                "Fingerprint Matrix Sparsity"
            ],
            "Value": [
                f"{num_drugs:,}",
                f"{avg_smiles_len:.2f} chars",
                f"{max_smiles_len:,} chars",
                f"{min_smiles_len:,} chars",
                f"{drug_fps.shape[0]} × {drug_fps.shape[1]}",
                f"{avg_active_bits:.2f} / 1024",
                f"{global_sparsity:.2%}"
            ]
        })

        table_preview = mo.ui.table(drug_struc_df_with_len.head(10), label="Preview of Drug Structures (Top 10)")

        return mo.vstack([
            mo.md("#### **Drug Structure & Fingerprint Summary Statistics**"),
            stats_df,
            mo.md("#### **Interactive Chemical Structures Preview**"),
            table_preview
        ])

    _display_drug_stats()
    return


@app.cell
def _():
    mo.md(r"""
    ### 1.3 Protein Sequence Mapping & Sequence Descriptors

    For target proteins, we map each unique target Gene/Protein identifier to its primary amino acid sequence and precomputed global sequence descriptors:

    1. **`989proseq.csv`**: Contains raw target protein ID-to-amino acid sequences.
    2. **`pro_ctd.txt`**: Contains 147-dimensional sequence descriptors representing **C**omposition (C), **T**ransition (T), and **D**istribution (D) of physicochemical properties.
    """)
    return


@app.cell
def _(dataset_root):
    pro_seq_df = pd.read_csv(dataset_root / "989proseq.csv")
    pro_seq_df.columns = ["pro_id", "pro_ids", "seq"]
    pro_ctds = np.loadtxt(dataset_root / "pro_ctd.txt", delimiter=",")
    return pro_ctds, pro_seq_df


@app.cell
def _(pro_ctds, pro_seq_df):
    def _display_protein_stats():
        # Compute protein lengths
        pro_seq_df_with_len = pro_seq_df.copy()
        pro_seq_df_with_len["seq_length"] = pro_seq_df_with_len["seq"].apply(len)

        num_proteins = len(pro_seq_df_with_len)
        avg_seq_len = pro_seq_df_with_len["seq_length"].mean()
        max_seq_len = pro_seq_df_with_len["seq_length"].max()
        min_seq_len = pro_seq_df_with_len["seq_length"].min()

        stats_df = pd.DataFrame({
            "Metric": [
                "Total Target Proteins",
                "Average Sequence Length",
                "Max Sequence Length",
                "Min Sequence Length",
                "CTD Descriptor Dimensions"
            ],
            "Value": [
                f"{num_proteins:,}",
                f"{avg_seq_len:.2f} residues",
                f"{max_seq_len:,} residues",
                f"{min_seq_len:,} residues",
                f"{pro_ctds.shape[0]} × {pro_ctds.shape[1]}"
            ]
        })

        table_preview = mo.ui.table(pro_seq_df_with_len[["pro_id", "seq", "seq_length"]].head(10), label="Preview of Protein Sequences (Top 10)")

        return mo.vstack([
            mo.md("#### **Protein Sequence & CTD Descriptor Summary Statistics**"),
            stats_df,
            mo.md("#### **Interactive Sequence Preview**"),
            table_preview
        ])

    _display_protein_stats()
    return


@app.cell
def _():
    mo.md(r"""
    ### 1.4 Combined Knowledge Graph Relation Triples

    The Yamanishi dataset provides two rich background Knowledge Graphs (KGs) to enable semantic link prediction and knowledge graph embeddings (KGE):

    1. **`kegg_kg.txt`**: Triples extracted from the KEGG database mapping relationships between pathways, genes, drug groups, ATC codes, and EC enzyme numbers.
    2. **`yamanishi_uniprot_kg.txt`**: Triples mapping relationships from UniProt, such as protein-protein interactions, protein-disease associations, and gene classifications.

    We merge both files to create a unified background Knowledge Graph. These triples are used downstream to train the PyKEEN DistMult model for semantic drug-target embedding representations.
    """)
    return


@app.cell
def _(dataset_root):
    # Load KEGG KG
    kegg_kg = pd.read_csv(
        dataset_root / "kg_data" / "kegg_kg.txt",
        sep="\t",
        header=None,
        names=["head", "relation", "tail"]
    ).astype(str)

    # Load Yamanishi UniProt KG
    uniprot_kg = pd.read_csv(
        dataset_root / "kg_data" / "yamanishi_uniprot_kg.txt",
        sep="\t",
        header=None,
        names=["head", "relation", "tail"]
    ).astype(str)

    # Combine them
    combined_kg = pd.concat([kegg_kg, uniprot_kg], ignore_index=True)
    return combined_kg, kegg_kg, uniprot_kg


@app.cell
def _(combined_kg, dti_df, kegg_kg, uniprot_kg):
    def _display_kg_stats():
        total_triples = len(combined_kg)
        num_kegg_triples = len(kegg_kg)
        num_uniprot_triples = len(uniprot_kg)

        unique_entities = pd.concat([combined_kg["head"], combined_kg["tail"]]).nunique()
        unique_relations = combined_kg["relation"].nunique()

        # DTI Entity Overlap
        dti_drugs = set(dti_df["drug_id"])
        dti_targets = set(dti_df["target_id"])

        kg_entities = set(pd.concat([combined_kg["head"], combined_kg["tail"]]))

        overlapping_drugs = dti_drugs.intersection(kg_entities)
        overlapping_targets = dti_targets.intersection(kg_entities)

        stats_df = pd.DataFrame({
            "Metric": [
                "Total Unified Triples",
                "KEGG KG Triples",
                "UniProt KG Triples",
                "Unique Entities (Nodes)",
                "Unique Relation Types (Edges)",
                "DTI Gold-Standard Drugs",
                "Drugs Present in KG (Coverage)",
                "DTI Gold-Standard Proteins",
                "Proteins Present in KG (Coverage)"
            ],
            "Value": [
                f"{total_triples:,}",
                f"{num_kegg_triples:,}",
                f"{num_uniprot_triples:,}",
                f"{unique_entities:,}",
                f"{unique_relations:,}",
                f"{len(dti_drugs):,}",
                f"{len(overlapping_drugs):,} ({len(overlapping_drugs)/len(dti_drugs):.2%})",
                f"{len(dti_targets):,}",
                f"{len(overlapping_targets):,} ({len(overlapping_targets)/len(dti_targets):.2%})"
            ]
        })

        table_preview = mo.ui.table(combined_kg.head(10), label="Preview of Background KG Triples (Top 10)")

        return mo.vstack([
            mo.md("#### **Knowledge Graph Summary Statistics & DTI Coverage**"),
            stats_df,
            mo.md("#### **Interactive Knowledge Graph Triples Preview**"),
            table_preview
        ])

    _display_kg_stats()
    return


@app.cell
def _(combined_kg):
    def _display_kg_plots():
        relation_counts = combined_kg["relation"].value_counts().reset_index()
        relation_counts.columns = ["relation", "count"]

        # Keep top 15 relations for readability in plot
        top_relations = relation_counts.head(15)

        # Styling matplotlib for dark/sleek theme
        plt.rcParams["figure.facecolor"] = "none"
        plt.rcParams["axes.facecolor"] = "none"
        plt.rcParams["text.color"] = "#E2E8F0"
        plt.rcParams["axes.labelcolor"] = "#94A3B8"
        plt.rcParams["xtick.color"] = "#94A3B8"
        plt.rcParams["ytick.color"] = "#94A3B8"
        plt.rcParams["grid.color"] = "#334155"
        plt.rcParams["axes.edgecolor"] = "#475569"

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Horizontal Bar Chart for Relations
        colors = plt.get_cmap("plasma")(np.linspace(0.2, 0.8, len(top_relations)))
        axes[0].barh(top_relations["relation"], top_relations["count"], color=colors, edgecolor="#1e1b4b", height=0.7)
        axes[0].invert_yaxis()  # top-down order
        axes[0].set_title("Top 15 KG Relation Frequencies", fontsize=12, fontweight="bold", pad=15, color="#F1F5F9")
        axes[0].set_xlabel("Frequency Count", fontsize=10)
        axes[0].grid(True, linestyle="--", alpha=0.3, axis="x")

        # Hub Degree distribution overview
        node_degrees = pd.concat([combined_kg["head"], combined_kg["tail"]]).value_counts()
        axes[1].hist(node_degrees, bins=50, color="#f59e0b", alpha=0.8, edgecolor="#78350f", rwidth=0.85, log=True)
        axes[1].set_title("KG Entity Connectivity Degree (Log Scale)", fontsize=12, fontweight="bold", pad=15, color="#F1F5F9")
        axes[1].set_xlabel("Degree (number of triples connected)", fontsize=10)
        axes[1].set_ylabel("Count of Entities (Log)", fontsize=10)
        axes[1].grid(True, linestyle="--", alpha=0.3)

        plt.tight_layout()
        plot_ui = mo.as_html(fig)
        plt.close(fig)

        # Top Hubs identification
        top_hubs = node_degrees.head(10).reset_index()
        top_hubs.columns = ["Entity ID", "Degree (KG Connections)"]
        hubs_table = mo.ui.table(top_hubs, label="Top 10 KG Entity Hubs")

        return mo.vstack([
            mo.md("#### **Knowledge Graph Connectivity & Topology**"),
            plot_ui,
            mo.md("#### **KG Hub Identification**"),
            hubs_table,
            mo.md(r"""
            #### **KG Structural Observations & ML Implications:**
            1. **Scale-Free Network Topology**:
               - The entity connectivity degree histogram (log-scale) shows that while most nodes have very low degrees (1 or 2 connections), a small number of hub nodes have several thousand connections. This is a classic scale-free network.
            2. **High DTI Entity Coverage**:
               - Over **99% of drugs** and **98% of target proteins** from the gold-standard interactions (`dt_all_08.txt`) are present in the combined background KG. This is crucial because it ensures that downstream KGE training (e.g. DistMult) will generate dense vector representations for almost all drug-target candidate pairs.
            """)
        ])

    _display_kg_plots()
    return


if __name__ == "__main__":
    app.run()
