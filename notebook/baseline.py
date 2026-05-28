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
    import random
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
def _(pro_ctds):
    _sample_ctd = pro_ctds[0]
    _c_part = _sample_ctd[0:21]
    _t_part = _sample_ctd[21:42]
    _d_part = _sample_ctd[42:147]

    mo.md(
        f"""
        To understand the 147-dimensional **CTD (Composition, Transition, and Distribution)** descriptor, let us examine the CTD representation for the first protein in our dataset (shape: `{_sample_ctd.shape}`).

        #### **1. Shape Breakdown**
        The 147 features represent three types of global sequence descriptors across 7 physical/chemical properties:
        *   **Composition ($C$ - indices 0 to 20; 21 dims):** The global percentage/fraction of amino acids belonging to a specific group (e.g., polar, neutral, hydrophobic) under 7 properties ($7 \\times 3 = 21$ dimensions).
        *   **Transition ($T$ - indices 21 to 41; 21 dims):** The transition frequency between adjacent amino acids of different groups ($7 \\times 3 = 21$ dimensions).
        *   **Distribution ($D$ - indices 42 to 146; 105 dims):** The relative positioning along the sequence ($0\\%$, $25\\%$, $50\\%$, $75\\%$, $100\\%$) for each group ($7 \\times 3 \\times 5 = 105$ dimensions).

        #### **2. Value Ranges & Numerical Observations**
        *   **Composition ($C$):** Range of values: `[{_c_part.min():.3f}, {_c_part.max():.3f}]`. These are fractions in the range $[0.0, 1.0]$.
        *   **Transition ($T$):** Range of values: `[{_t_part.min():.3f}, {_t_part.max():.3f}]`. These are transition frequencies/probabilities, also in the range $[0.0, 1.0]$.
        *   **Distribution ($D$):** Range of values: `[{_d_part.min():.3f}, {_d_part.max():.3f}]`. These represent position percentages along the primary sequence and are defined in the range $[0.0, 100.0]$.

        > **Scale Discrepancy:** The $C$ and $T$ features are in $[0, 1]$, whereas the $D$ features are in $[0, 100]$. Because of this 100x scale discrepancy, applying a scaler like `MinMaxScaler` or `StandardScaler` is absolutely vital before passing these features to distance-based or gradient-descent algorithms (such as Neural Networks or SVMs), even though Random Forest handles it naturally.
        """
    )
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 2. Establishing a Random Forest Baseline

    Random Forest is suitable for this problem because:
    - **Highly Suitable for Tabular Bio-descriptors**: Random Forest naturally handles high-dimensional, sparse binary features (like 1024-bit Morgan drug fingerprints) and continuous features (like 147-dimensional CTD protein descriptors) without requiring complex neural architectures.
    - **Simplicity**: Treats DTI prediction as an independent and identically distributed (i.i.d.) tabular classification task, validating our pipeline and feature alignment logic.

    It serves as a benchmark for comparison before introducing graph embedding or deep learning models.

    ## 2.1 Simple Train-Test Split (80/20)

    ### Dataset Reconstruction & Model Input Format
    We construct a balanced or realistically ratioed tabular dataset where **each row in the DataFrame represents a single drug-target pair prediction instance**:
    - **Positive Instances (`label = 1.0`)**: 5,128 verified interactions loaded from `dt_all_08.txt`.
    - **Negative Instances (`label = 0.0`)**: 46,152 programmatically sampled unobserved pairs (maintaining a realistic **1:9 positive-to-negative ratio**).
    - **Final Features DataFrame (`X`)**: A `pd.DataFrame` of shape `(51280, 1171)` containing concatenated drug Morgan fingerprints (1024 cols) and scaled target CTD descriptors (147 cols).
    - **Target Series (`y`)**: A `pd.Series` of shape `(51280,)` with binary labels.
    """)
    return


@app.cell
def _(drug_struc_df, dti_df, pro_seq_df):
    # 1. Positive pair
    _pos_pairs = set(zip(dti_df["drug_id"], dti_df["target_id"]))

    # 2. Extract all unique drug and target IDs that have features
    _all_drugs = list(drug_struc_df["drug_id"].unique())
    _all_targets = list(pro_seq_df["pro_id"].unique())

    # 3. Programmatic negative sampling (1:9 ratio)
    _num_positives = len(_pos_pairs)
    _num_negatives = _num_positives * 9

    random.seed(42)
    _neg_pairs_set = set()
    while len(_neg_pairs_set) < _num_negatives:
        drug = random.choice(_all_drugs)
        target = random.choice(_all_targets)
        pair = (drug, target)
        if pair not in _pos_pairs and pair not in _neg_pairs_set:
            _neg_pairs_set.add(pair)
    _neg_pairs = list(_neg_pairs_set)

    # 4. Combine into a reconstructed DTI DataFrame
    _pos_df = pd.DataFrame(list(_pos_pairs), columns=["drug_id", "target_id"])
    _pos_df["label"] = 1.0

    _neg_df = pd.DataFrame(_neg_pairs, columns=["drug_id", "target_id"])
    _neg_df["label"] = 0.0

    data_df = pd.concat([_pos_df, _neg_df], ignore_index=True)
    return (data_df,)


@app.cell
def _(data_df, drug_fps, drug_struc_df, pro_ctds, pro_seq_df):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split

    # 1. Scale target features using MinMaxScaler (as in kge_rf.py)
    _scaler = MinMaxScaler(feature_range=(0, 1))
    _scaled_pro_ctds = _scaler.fit_transform(pro_ctds)

    # 2. Build feature lookup maps/DataFrames (explicit prefixes to avoid col conflicts)
    _drug_id_series = drug_struc_df["drug_id"]
    _fp_cols = [f"drug_fp_{i}" for i in range(drug_fps.shape[1])]
    _fp_df = pd.concat([_drug_id_series, pd.DataFrame(drug_fps, columns=_fp_cols)], axis=1)

    _pro_id_series = pro_seq_df["pro_id"]
    _ctd_cols = [f"pro_ctd_{i}" for i in range(_scaled_pro_ctds.shape[1])]
    _ctd_df = pd.concat([_pro_id_series, pd.DataFrame(_scaled_pro_ctds, columns=_ctd_cols)], axis=1)

    # 3. Merge features to reconstructed DTI DataFrame
    _merged_drug = pd.merge(data_df, _fp_df, how="left", on="drug_id")
    _merged_all = pd.merge(
        _merged_drug,
        _ctd_df,
        how="left",
        left_on="target_id",
        right_on="pro_id",
    )

    # 4. Extract target label and features DataFrame
    _X = _merged_all.drop(columns=["drug_id", "target_id", "label", "pro_id"])
    _y = _merged_all["label"]

    # 5. Perform standard stratified 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        _X, _y, test_size=0.2, random_state=42, stratify=_y
    )

    _slice_cols = lambda df: pd.concat([df.iloc[:, :6], df.iloc[:, -6:]], axis=1)

    mo.vstack([
        mo.md(f"### Dataset Overview"),
        mo.md(f"Shape of X: {_X.shape}"),
        mo.md(f"Samples of X (5, showing first/last 6 columns):"),
        _slice_cols(_X).sample(5, random_state=42),
        mo.md(f"Shape of y: {_y.shape}"),
        mo.md(f"Samples of y (5):"),
        _y.sample(5, random_state=42),
        mo.md(f"Shape of X_train: {X_train.shape}"),
        mo.md(f"Samples of X_train (5, showing first/last 6 columns):"),
        _slice_cols(X_train).sample(5, random_state=42),
        mo.md(f"Shape of X_test: {X_test.shape}"),
        mo.md(f"Samples of X_test (5, showing first/last 6 columns):"),
        _slice_cols(X_test).sample(5, random_state=42),
        mo.md(f"Shape of y_train: {y_train.shape}"),
        mo.md(f"Distribution of y_train:"),
        y_train.value_counts(normalize=True),
        mo.md(f"Shape of y_test: {y_test.shape}"),
        mo.md(f"Distribution of y_test:"),
        y_test.value_counts(normalize=True),
    ])
    return X_test, X_train, y_test, y_train


@app.cell
def _(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier

    # 1. Initialize and train the Random Forest Classifier
    random_forest_clf = RandomForestClassifier(
        n_estimators=200,
        criterion='entropy',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    random_forest_clf.fit(X_train, y_train)

    mo.md("")
    return (random_forest_clf,)


@app.cell
def _(X_test, random_forest_clf, y_test):
    from sklearn import metrics

    # 2. Predict probabilities on the test set
    _y_pred_proba = random_forest_clf.predict_proba(X_test)[:, 1]

    # 3. Calculate ROC-AUC and PR-AUC
    _fpr, _tpr, _ = metrics.roc_curve(y_test, _y_pred_proba)
    _roc_auc = metrics.auc(_fpr, _tpr)

    _precision, _recall, _ = metrics.precision_recall_curve(y_test, _y_pred_proba)
    _pr_auc = metrics.auc(_recall, _precision)

    # 4. Display results
    _stats_df = pd.DataFrame({
        "Metric": ["ROC-AUC", "Precision-Recall AUC (PR-AUC)"],
        "Value": [f"{_roc_auc:.4f}", f"{_pr_auc:.4f}"]
    })

    # Sleek dark/modern theme styling for matplotlib
    plt.rcParams["figure.facecolor"] = "none"
    plt.rcParams["axes.facecolor"] = "none"
    plt.rcParams["text.color"] = "#E2E8F0"
    plt.rcParams["axes.labelcolor"] = "#94A3B8"
    plt.rcParams["xtick.color"] = "#94A3B8"
    plt.rcParams["ytick.color"] = "#94A3B8"
    plt.rcParams["grid.color"] = "#334155"
    plt.rcParams["axes.edgecolor"] = "#475569"

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Plot ROC Curve
    _axes[0].plot(_fpr, _tpr, color="#8b5cf6", lw=2.5, label=f"ROC Curve (AUC = {_roc_auc:.4f})")
    _axes[0].plot([0, 1], [0, 1], color="#94A3B8", linestyle="--", alpha=0.5)
    _axes[0].set_xlim([0.0, 1.0])
    _axes[0].set_ylim([0.0, 1.05])
    _axes[0].set_xlabel("False Positive Rate (FPR)", fontsize=10)
    _axes[0].set_ylabel("True Positive Rate (TPR)", fontsize=10)
    _axes[0].set_title("Receiver Operating Characteristic (ROC)", fontsize=11, fontweight="bold", pad=12, color="#F1F5F9")
    _axes[0].legend(loc="lower right", facecolor="#1e293b", edgecolor="#475569")
    _axes[0].grid(True, linestyle="--", alpha=0.3)

    # 2. Plot Precision-Recall Curve
    _axes[1].plot(_recall, _precision, color="#d946ef", lw=2.5, label=f"PR Curve (AUC = {_pr_auc:.4f})")
    _axes[1].set_xlim([0.0, 1.0])
    _axes[1].set_ylim([0.0, 1.05])
    _axes[1].set_xlabel("Recall", fontsize=10)
    _axes[1].set_ylabel("Precision", fontsize=10)
    _axes[1].set_title("Precision-Recall (PR) Curve", fontsize=11, fontweight="bold", pad=12, color="#F1F5F9")
    _axes[1].legend(loc="lower left", facecolor="#1e293b", edgecolor="#475569")
    _axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    _plot_ui = mo.as_html(_fig)
    plt.close(_fig)

    # Result Summary
    _result_summary = mo.md(
        f"""
        #### **Result Interpretation & Validation Observations**

        1. **Outstanding Baseline Performance**:
            *   **ROC-AUC ($0.9448$):** Demonstrates excellent discriminative ability between interacting and non-interacting drug-target pairs across all classification thresholds.
            *   **PR-AUC ($0.8091$):** Since the dataset is highly imbalanced ($1:9$ positive-to-negative ratio), the Precision-Recall AUC is a much more robust and honest indicator of performance. Achieving an $0.8091$ PR-AUC indicates the model maintains high precision (low false-positive rate) even at high recall thresholds.

        2. **Why is the baseline so strong?**
            *   **Highly Informative Features:** Circular Morgan fingerprints represent local chemical neighborhoods that directly dictate binding affinity, while 147-dimensional CTD descriptors represent global physical/chemical signatures of target proteins.
            *   **Warm-Start Bipartite Split:** This standard $80/20$ split evaluates the model in a "warm-start" setting where the individual drugs and target proteins in the test set have been seen during training, just in different pairings. This is expected to yield very high baseline performance. In a "cold-start" setting (unseen drugs or unseen proteins), performance would be lower.
        """
    )

    mo.vstack([
        mo.md("### 2.1.1 Random Forest Baseline Metrics"),
        _stats_df,
        _plot_ui,
        _result_summary,
    ])
    return


if __name__ == "__main__":
    app.run()
