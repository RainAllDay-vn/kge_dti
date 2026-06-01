# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "scikit-learn",
#     "xgboost",
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
    import pickle

    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics

    random.seed(42)


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
    # Set global variables
    dataset_root = Path("./data/yamanishi_08")
    models_dir = Path("./models")
    models_dir.mkdir(parents=True, exist_ok=True)
    return dataset_root, models_dir


@app.cell
def _(ctd_df, dataset_root, fp_df, models_dir):
    def evaluate_baseline_cv(model_class, model_kwargs, prefix):
        results = {}
        splits = ["warm_start_1_10", "warm_start_1_1", "protein_coldstart", "drug_coldstart"]

        for split in splits:
            cv_metrics_path = models_dir / f"baseline_cv_metrics_{prefix}_{split}.pkl"

            if cv_metrics_path.exists():
                with open(cv_metrics_path, "rb") as f:
                    results[split] = pickle.load(f)
            else:
                # Also check joint pickle if it exists and contains the prefix key
                joint_path = models_dir / f"baseline_cv_metrics_{split}.pkl"
                if joint_path.exists():
                    with open(joint_path, "rb") as f:
                        data = pickle.load(f)
                        if isinstance(data, dict) and prefix in data:
                            results[split] = data[prefix]
                            continue
                        elif isinstance(data, dict) and prefix in ["rf", "xgb"] and "roc_auc" in data:
                            results[split] = data
                            continue

                results[split] = {"roc_auc": [], "pr_auc": []}

                # Load and train over 10 folds
                for fold in range(10):
                    train_path = dataset_root / "data_folds" / split / f"train_fold_{fold + 1}.csv"
                    test_path = dataset_root / "data_folds" / split / f"test_fold_{fold + 1}.csv"

                    train_df = pd.read_csv(train_path)
                    test_df = pd.read_csv(test_path)

                    def prepare_fold_features(df):
                        merged = pd.merge(df, fp_df, how="left", left_on="head", right_on="drug_id")
                        merged = pd.merge(merged, ctd_df, how="left", left_on="tail", right_on="pro_id")
                        X = merged.drop(columns=["head", "relation", "tail", "label", "drug_id", "pro_id", "pred"], errors="ignore")
                        y = merged["label"]
                        return X, y

                    X_train, y_train = prepare_fold_features(train_df)
                    X_test, y_test = prepare_fold_features(test_df)

                    # Dynamic kwargs copy to avoid mutating the original
                    current_kwargs = model_kwargs.copy()
                    if model_class.__name__ == "XGBClassifier":
                        num_pos = (y_train == 1.0).sum()
                        num_neg = (y_train == 0.0).sum()
                        ratio = num_neg / num_pos if num_pos > 0 else 1.0
                        current_kwargs["scale_pos_weight"] = ratio

                    model = model_class(**current_kwargs)
                    model.fit(X_train, y_train)
                    probs = model.predict_proba(X_test)[:, 1]

                    fpr, tpr, _ = metrics.roc_curve(y_test, probs)
                    roc_auc = metrics.auc(fpr, tpr)

                    prec, rec, _ = metrics.precision_recall_curve(y_test, probs)
                    pr_auc = metrics.auc(rec, prec)

                    results[split]["roc_auc"].append(roc_auc)
                    results[split]["pr_auc"].append(pr_auc)

                with open(cv_metrics_path, "wb") as f:
                    pickle.dump(results[split], f)

        return results

    return (evaluate_baseline_cv,)


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
    ### 1.1 Drug-Target Interactions

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
def _():
    mo.md(r"""
    #### DTI Dataset Summary Statistics
    """)
    return


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

        return stats_df

    _display_stats()
    return


@app.cell
def _():
    mo.md(r"""
    #### 1.1.1 Degree Distributions & DTI Network Sparsity

    To understand the connectivity landscape, we analyze the degree distribution of the bipartite DTI network. The degree of a drug represents the number of target proteins it is known to interact with, while the degree of a protein represents the number of active drugs targeting it.
    """)
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

        return plot_ui

    _display_degree_stats()
    return


@app.cell
def _():
    mo.md(r"""
    ##### **Key Insights & Structural Observations:**

    1. **Low-Degree Node Domination (High Sparsity)**:
        - **Drugs**: Nearly **74.6% of all drugs** (590 out of 791) have a degree of 5 or fewer, meaning the vast majority of drugs are associated with very few targets.
        - **Targets**: More than **54.6% of all target proteins** (540 out of 989) interact with 2 or fewer drugs.
    2. **Power-Law/Scale-Free Characteristics**:
        - Both distributions exhibit a heavy-tailed decay. A tiny group of highly connected "hub" nodes (e.g., a few drugs targeting over 100 proteins) hold the network together, while most nodes have very sparse connections.
    3. **Interaction Matrix Density ($0.655\%$)**:
        - Out of $782,301$ possible bipartite edges ($791 \text{ drugs} \times 989 \text{ targets}$), only **5,128** are active. This extreme sparsity ($99.345\%$ empty space) presents a classic sparse matrix completion challenge for downstream DTI machine learning models.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    #### DTI Network Hub Identification
    Below are the most interactive entities (hubs) in the DTI network:
    """)
    return


@app.cell
def _(dti_df):
    def _display_hub_view():
        top_drugs = dti_df.groupby("drug_id").size().reset_index(name="degree").sort_values(by="degree", ascending=False).head(10)
        top_targets = dti_df.groupby("target_id").size().reset_index(name="degree").sort_values(by="degree", ascending=False).head(10)

        top_drugs_table = mo.ui.table(top_drugs, label="Top 10 Most Interactive Drugs")
        top_targets_table = mo.ui.table(top_targets, label="Top 10 Most Interactive Targets")

        hub_view = mo.hstack([top_drugs_table, top_targets_table], justify="space-around")

        return hub_view

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
def _():
    mo.md(r"""
    #### Drug Structure & Fingerprint Summary Statistics
    """)
    return


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

        return stats_df

    _display_drug_stats()
    return


@app.cell
def _():
    mo.md(r"""
    #### Interactive Chemical Structures Preview
    """)
    return


@app.cell
def _(drug_struc_df):
    _df = drug_struc_df.copy()
    _df["smiles_length"] = _df["smiles"].apply(len)
    mo.ui.table(_df.head(10), label="Preview of Drug Structures (Top 10)")
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
def _():
    mo.md(r"""
    #### Protein Sequence & CTD Descriptor Summary Statistics
    """)
    return


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

        return stats_df

    _display_protein_stats()
    return


@app.cell
def _():
    mo.md(r"""
    #### Interactive Sequence Preview
    """)
    return


@app.cell
def _(pro_seq_df):
    _df = pro_seq_df.copy()
    _df["seq_length"] = _df["seq"].apply(len)
    mo.ui.table(_df[["pro_id", "seq", "seq_length"]].head(10), label="Preview of Protein Sequences (Top 10)")
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
def _():
    mo.md(r"""
    #### Knowledge Graph Summary Statistics & DTI Coverage
    """)
    return


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

        return stats_df

    _display_kg_stats()
    return


@app.cell
def _():
    mo.md(r"""
    #### Interactive Knowledge Graph Triples Preview
    """)
    return


@app.cell
def _(combined_kg):
    table_preview = mo.ui.table(combined_kg.head(10), label="Preview of Background KG Triples (Top 10)")
    table_preview
    return


@app.cell
def _():
    mo.md(r"""
    #### Knowledge Graph Connectivity & Topology
    """)
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

        return plot_ui

    plot_ui = _display_kg_plots()
    plot_ui
    return


@app.cell
def _():
    mo.md(r"""
    #### KG Hub Identification
    """)
    return


@app.cell
def _(combined_kg):
    node_degrees = pd.concat([combined_kg["head"], combined_kg["tail"]]).value_counts()
    top_hubs = node_degrees.head(10).reset_index()
    top_hubs.columns = ["Entity ID", "Degree (KG Connections)"]
    hubs_table = mo.ui.table(top_hubs, label="Top 10 KG Entity Hubs")
    hubs_table
    return


@app.cell
def _():
    mo.md(r"""
    ##### **KG Structural Observations & ML Implications:**

    1. **Scale-Free Network Topology**:
       - The entity connectivity degree histogram (log-scale) shows that while most nodes have very low degrees (1 or 2 connections), a small number of hub nodes have several thousand connections. This is a classic scale-free network.
    2. **High DTI Entity Coverage**:
       - Over **99% of drugs** and **98% of target proteins** from the gold-standard interactions (`dt_all_08.txt`) are present in the combined background KG. This is crucial because it ensures that downstream KGE training (e.g. DistMult) will generate dense vector representations for almost all drug-target candidate pairs.
    """)
    return


@app.cell
def _(drug_fps, drug_struc_df, pro_ctds, pro_seq_df):
    from sklearn.preprocessing import MinMaxScaler

    # 1. Scale target features using MinMaxScaler
    _scaler = MinMaxScaler(feature_range=(0, 1))
    _scaled_pro_ctds = _scaler.fit_transform(pro_ctds)

    # 2. Build feature lookup maps/DataFrames (explicit prefixes to avoid col conflicts)
    _drug_id_series = drug_struc_df["drug_id"]
    _fp_cols = [f"drug_fp_{i}" for i in range(drug_fps.shape[1])]
    fp_df = pd.concat([_drug_id_series, pd.DataFrame(drug_fps, columns=_fp_cols)], axis=1)

    _pro_id_series = pro_seq_df["pro_id"]
    _ctd_cols = [f"pro_ctd_{i}" for i in range(_scaled_pro_ctds.shape[1])]
    ctd_df = pd.concat([_pro_id_series, pd.DataFrame(_scaled_pro_ctds, columns=_ctd_cols)], axis=1)
    return ctd_df, fp_df


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. KNN Baseline Model

    In this section, we evaluate the **KNN Classifier** baseline across four dataset partitions:
    - **Warm-Start 1:10 Split:** Bipartite random split with 1:10 positive-to-negative ratio.
    - **Warm-Start 1:1 Split:** Bipartite random split with 1:1 positive-to-negative ratio.
    - **Protein Cold-Start Split:** Predicting on novel, unseen target proteins.
    - **Drug Cold-Start Split:** Predicting on novel, unseen drug structures.

    We utilize the ready-made 10-fold cross-validation folds in `data/yamanishi_08/data_folds/` to train and evaluate the models.
    """)
    return


@app.cell
def _(evaluate_baseline_cv):
    cv_results_knn = evaluate_baseline_cv(
        KNeighborsClassifier,
        {"n_neighbors": 5, "n_jobs": -1},
        "knn"
    )
    return (cv_results_knn,)


@app.cell
def _():
    mo.md(r"""
    ### 2.1 Cross-Validation Results
    """)
    return


@app.cell
def _(cv_results_knn):
    _splits = ["warm_start_1_10", "warm_start_1_1", "protein_coldstart", "drug_coldstart"]
    _split_labels = ["Warm-Start 1:10", "Warm-Start 1:1", "Protein Cold-Start", "Drug Cold-Start"]

    _rows = []
    for _s, _label in zip(_splits, _split_labels):
        _roc_mean = np.mean(cv_results_knn[_s]["roc_auc"])
        _roc_std = np.std(cv_results_knn[_s]["roc_auc"])
        _pr_mean = np.mean(cv_results_knn[_s]["pr_auc"])
        _pr_std = np.std(cv_results_knn[_s]["pr_auc"])
        _rows.append({
            "Evaluation Split": _label,
            "Mean ROC-AUC": f"{_roc_mean:.4f} (± {_roc_std:.4f})",
            "Mean PR-AUC": f"{_pr_mean:.4f} (± {_pr_std:.4f})"
        })

    _df = pd.DataFrame(_rows)

    plt.rcParams["figure.facecolor"] = "none"
    plt.rcParams["axes.facecolor"] = "none"
    plt.rcParams["text.color"] = "#E2E8F0"
    plt.rcParams["axes.labelcolor"] = "#94A3B8"
    plt.rcParams["xtick.color"] = "#94A3B8"
    plt.rcParams["ytick.color"] = "#94A3B8"
    plt.rcParams["grid.color"] = "#334155"
    plt.rcParams["axes.edgecolor"] = "#475569"

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

    _knn_roc_data = [cv_results_knn[s]["roc_auc"] for s in _splits]
    _knn_pr_data = [cv_results_knn[s]["pr_auc"] for s in _splits]

    _axes[0].boxplot(_knn_roc_data, labels=_split_labels)
    _axes[0].set_title("KNN ROC-AUC", fontsize=11, fontweight="bold", pad=12, color="#F1F5F9")
    _axes[0].set_ylabel("ROC-AUC Score", fontsize=10)
    _axes[0].grid(True, linestyle="--", alpha=0.3)

    _axes[1].boxplot(_knn_pr_data, labels=_split_labels)
    _axes[1].set_title("KNN PR-AUC", fontsize=11, fontweight="bold", pad=12, color="#F1F5F9")
    _axes[1].set_ylabel("PR-AUC Score", fontsize=10)
    _axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    _plot = mo.as_html(_fig)
    plt.close(_fig)

    mo.vstack([
        _df,
        _plot
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Logistic Regression Baseline Model

    In this section, we evaluate the **Logistic Regression** baseline across four dataset partitions:
    - **Warm-Start 1:10 Split:** Bipartite random split with 1:10 positive-to-negative ratio.
    - **Warm-Start 1:1 Split:** Bipartite random split with 1:1 positive-to-negative ratio.
    - **Protein Cold-Start Split:** Predicting on novel, unseen target proteins.
    - **Drug Cold-Start Split:** Predicting on novel, unseen drug structures.

    We utilize the ready-made 10-fold cross-validation folds in `data/yamanishi_08/data_folds/` to train and evaluate the models.
    """)
    return


@app.cell
def _(evaluate_baseline_cv):
    cv_results_lr = evaluate_baseline_cv(
        LogisticRegression,
        {"max_iter": 1000, "class_weight": "balanced", "random_state": 42},
        "lr"
    )
    return (cv_results_lr,)


@app.cell
def _():
    mo.md(r"""
    ### 3.1 Cross-Validation Results
    """)
    return


@app.cell
def _(cv_results_lr):
    _splits = ["warm_start_1_10", "warm_start_1_1", "protein_coldstart", "drug_coldstart"]
    _split_labels = ["Warm-Start 1:10", "Warm-Start 1:1", "Protein Cold-Start", "Drug Cold-Start"]

    _rows = []
    for _s, _label in zip(_splits, _split_labels):
        _roc_mean = np.mean(cv_results_lr[_s]["roc_auc"])
        _roc_std = np.std(cv_results_lr[_s]["roc_auc"])
        _pr_mean = np.mean(cv_results_lr[_s]["pr_auc"])
        _pr_std = np.std(cv_results_lr[_s]["pr_auc"])
        _rows.append({
            "Evaluation Split": _label,
            "Mean ROC-AUC": f"{_roc_mean:.4f} (± {_roc_std:.4f})",
            "Mean PR-AUC": f"{_pr_mean:.4f} (± {_pr_std:.4f})"
        })

    _df = pd.DataFrame(_rows)

    plt.rcParams["figure.facecolor"] = "none"
    plt.rcParams["axes.facecolor"] = "none"
    plt.rcParams["text.color"] = "#E2E8F0"
    plt.rcParams["axes.labelcolor"] = "#94A3B8"
    plt.rcParams["xtick.color"] = "#94A3B8"
    plt.rcParams["ytick.color"] = "#94A3B8"
    plt.rcParams["grid.color"] = "#334155"
    plt.rcParams["axes.edgecolor"] = "#475569"

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

    _lr_roc_data = [cv_results_lr[s]["roc_auc"] for s in _splits]
    _lr_pr_data = [cv_results_lr[s]["pr_auc"] for s in _splits]

    _axes[0].boxplot(_lr_roc_data, labels=_split_labels)
    _axes[0].set_title("Logistic Regression ROC-AUC", fontsize=11, fontweight="bold", pad=12, color="#F1F5F9")
    _axes[0].set_ylabel("ROC-AUC Score", fontsize=10)
    _axes[0].grid(True, linestyle="--", alpha=0.3)

    _axes[1].boxplot(_lr_pr_data, labels=_split_labels)
    _axes[1].set_title("Logistic Regression PR-AUC", fontsize=11, fontweight="bold", pad=12, color="#F1F5F9")
    _axes[1].set_ylabel("PR-AUC Score", fontsize=10)
    _axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    _plot = mo.as_html(_fig)
    plt.close(_fig)

    mo.vstack([
        _df,
        _plot
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Random Forest Baseline Model

    In this section, we evaluate the **Random Forest Classifier** baseline across four dataset partitions:
    - **Warm-Start 1:10 Split:** Bipartite random split with 1:10 positive-to-negative ratio.
    - **Warm-Start 1:1 Split:** Bipartite random split with 1:1 positive-to-negative ratio.
    - **Protein Cold-Start Split:** Predicting on novel, unseen target proteins.
    - **Drug Cold-Start Split:** Predicting on novel, unseen drug structures.

    We utilize the ready-made 10-fold cross-validation folds in `data/yamanishi_08/data_folds/` to train and evaluate the models.
    """)
    return


@app.cell
def _(evaluate_baseline_cv):
    cv_results_rf = evaluate_baseline_cv(
        RandomForestClassifier,
        {
            "n_estimators": 200,
            "criterion": "entropy",
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1
        },
        "rf"
    )
    return (cv_results_rf,)


@app.cell
def _():
    mo.md(r"""
    ### 4.1 Cross-Validation Results
    """)
    return


@app.cell
def _(cv_results_rf):
    _splits = ["warm_start_1_10", "warm_start_1_1", "protein_coldstart", "drug_coldstart"]
    _split_labels = ["Warm-Start 1:10", "Warm-Start 1:1", "Protein Cold-Start", "Drug Cold-Start"]

    _rows = []
    for _s, _label in zip(_splits, _split_labels):
        _roc_mean = np.mean(cv_results_rf[_s]["roc_auc"])
        _roc_std = np.std(cv_results_rf[_s]["roc_auc"])
        _pr_mean = np.mean(cv_results_rf[_s]["pr_auc"])
        _pr_std = np.std(cv_results_rf[_s]["pr_auc"])
        _rows.append({
            "Evaluation Split": _label,
            "Mean ROC-AUC": f"{_roc_mean:.4f} (± {_roc_std:.4f})",
            "Mean PR-AUC": f"{_pr_mean:.4f} (± {_pr_std:.4f})"
        })

    _df = pd.DataFrame(_rows)

    plt.rcParams["figure.facecolor"] = "none"
    plt.rcParams["axes.facecolor"] = "none"
    plt.rcParams["text.color"] = "#E2E8F0"
    plt.rcParams["axes.labelcolor"] = "#94A3B8"
    plt.rcParams["xtick.color"] = "#94A3B8"
    plt.rcParams["ytick.color"] = "#94A3B8"
    plt.rcParams["grid.color"] = "#334155"
    plt.rcParams["axes.edgecolor"] = "#475569"

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

    _rf_roc_data = [cv_results_rf[s]["roc_auc"] for s in _splits]
    _rf_pr_data = [cv_results_rf[s]["pr_auc"] for s in _splits]

    _axes[0].boxplot(_rf_roc_data, labels=_split_labels)
    _axes[0].set_title("Random Forest ROC-AUC", fontsize=11, fontweight="bold", pad=12, color="#F1F5F9")
    _axes[0].set_ylabel("ROC-AUC Score", fontsize=10)
    _axes[0].grid(True, linestyle="--", alpha=0.3)

    _axes[1].boxplot(_rf_pr_data, labels=_split_labels)
    _axes[1].set_title("Random Forest PR-AUC", fontsize=11, fontweight="bold", pad=12, color="#F1F5F9")
    _axes[1].set_ylabel("PR-AUC Score", fontsize=10)
    _axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    _plot = mo.as_html(_fig)
    plt.close(_fig)

    mo.vstack([
        _df,
        _plot
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. XGBoost Baseline Model

    In this section, we evaluate the **XGBoost Classifier** baseline across four dataset partitions:
    - **Warm-Start 1:10 Split:** Bipartite random split with 1:10 positive-to-negative ratio.
    - **Warm-Start 1:1 Split:** Bipartite random split with 1:1 positive-to-negative ratio.
    - **Protein Cold-Start Split:** Predicting on novel, unseen target proteins.
    - **Drug Cold-Start Split:** Predicting on novel, unseen drug structures.

    Like Random Forest, we utilize the ready-made 10-fold cross-validation folds in `data/yamanishi_08/data_folds/` to train and evaluate the models.
    """)
    return


@app.cell
def _(evaluate_baseline_cv):
    cv_results_xgb = evaluate_baseline_cv(
        XGBClassifier,
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1
        },
        "xgb"
    )
    return (cv_results_xgb,)


@app.cell
def _():
    mo.md(r"""
    ### 5.1 Cross-Validation Results
    """)
    return


@app.cell
def _(cv_results_xgb):
    _splits = ["warm_start_1_10", "warm_start_1_1", "protein_coldstart", "drug_coldstart"]
    _split_labels = ["Warm-Start 1:10", "Warm-Start 1:1", "Protein Cold-Start", "Drug Cold-Start"]

    _rows = []
    for _s, _label in zip(_splits, _split_labels):
        _roc_mean = np.mean(cv_results_xgb[_s]["roc_auc"])
        _roc_std = np.std(cv_results_xgb[_s]["roc_auc"])
        _pr_mean = np.mean(cv_results_xgb[_s]["pr_auc"])
        _pr_std = np.std(cv_results_xgb[_s]["pr_auc"])
        _rows.append({
            "Evaluation Split": _label,
            "Mean ROC-AUC": f"{_roc_mean:.4f} (± {_roc_std:.4f})",
            "Mean PR-AUC": f"{_pr_mean:.4f} (± {_pr_std:.4f})"
        })

    _df = pd.DataFrame(_rows)

    plt.rcParams["figure.facecolor"] = "none"
    plt.rcParams["axes.facecolor"] = "none"
    plt.rcParams["text.color"] = "#E2E8F0"
    plt.rcParams["axes.labelcolor"] = "#94A3B8"
    plt.rcParams["xtick.color"] = "#94A3B8"
    plt.rcParams["ytick.color"] = "#94A3B8"
    plt.rcParams["grid.color"] = "#334155"
    plt.rcParams["axes.edgecolor"] = "#475569"

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

    _xgb_roc_data = [cv_results_xgb[s]["roc_auc"] for s in _splits]
    _xgb_pr_data = [cv_results_xgb[s]["pr_auc"] for s in _splits]

    _axes[0].boxplot(_xgb_roc_data, labels=_split_labels)
    _axes[0].set_title("XGBoost ROC-AUC", fontsize=11, fontweight="bold", pad=12, color="#F1F5F9")
    _axes[0].set_ylabel("ROC-AUC Score", fontsize=10)
    _axes[0].grid(True, linestyle="--", alpha=0.3)

    _axes[1].boxplot(_xgb_pr_data, labels=_split_labels)
    _axes[1].set_title("XGBoost PR-AUC", fontsize=11, fontweight="bold", pad=12, color="#F1F5F9")
    _axes[1].set_ylabel("PR-AUC Score", fontsize=10)
    _axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    _plot = mo.as_html(_fig)
    plt.close(_fig)

    mo.vstack([
        _df,
        _plot
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6. Model Comparison & Conclusion

    In this section, we compare **Random Forest** and **XGBoost** side-by-side across all four cross-validation configurations to determine generalisation patterns, strengths, and bottlenecks.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### 6.1 Performance Comparison across Splits
    """)
    return


@app.cell
def _(cv_results_knn, cv_results_lr, cv_results_rf, cv_results_xgb):
    # Styling matplotlib for dark/sleek theme
    plt.rcParams["figure.facecolor"] = "none"
    plt.rcParams["axes.facecolor"] = "none"
    plt.rcParams["text.color"] = "#E2E8F0"
    plt.rcParams["axes.labelcolor"] = "#94A3B8"
    plt.rcParams["xtick.color"] = "#94A3B8"
    plt.rcParams["ytick.color"] = "#94A3B8"
    plt.rcParams["grid.color"] = "#334155"
    plt.rcParams["axes.edgecolor"] = "#475569"

    _fig, _axes = plt.subplots(1, 2, figsize=(18, 7))

    _splits = ["warm_start_1_10", "warm_start_1_1", "protein_coldstart", "drug_coldstart"]
    _split_labels = ["Warm-Start 1:10", "Warm-Start 1:1", "Protein Cold-Start", "Drug Cold-Start"]

    _positions_knn = [1, 6, 11, 16]
    _positions_lr = [2, 7, 12, 17]
    _positions_rf = [3, 8, 13, 18]
    _positions_xgb = [4, 9, 14, 19]

    # Customise box plots
    def _style_boxplots(bp, color):
        for box in bp['boxes']:
            box.set(color=color, linewidth=2)
        for whisker in bp['whiskers']:
            whisker.set(color="#94A3B8", linewidth=1.5, linestyle="--")
        for cap in bp['caps']:
            cap.set(color="#94A3B8", linewidth=1.5)
        for median in bp['medians']:
            median.set(color="#f59e0b", linewidth=2.5)
        for flier in bp['fliers']:
            flier.set(marker='o', color='#ef4444', alpha=0.8)

    # 1. ROC-AUC Grouped Boxplot
    _knn_roc_data = [cv_results_knn[s]["roc_auc"] for s in _splits]
    _lr_roc_data = [cv_results_lr[s]["roc_auc"] for s in _splits]
    _rf_roc_data = [cv_results_rf[s]["roc_auc"] for s in _splits]
    _xgb_roc_data = [cv_results_xgb[s]["roc_auc"] for s in _splits]

    _bp_knn_roc = _axes[0].boxplot(_knn_roc_data, positions=_positions_knn, widths=0.4, patch_artist=False)
    _bp_lr_roc = _axes[0].boxplot(_lr_roc_data, positions=_positions_lr, widths=0.4, patch_artist=False)
    _bp_rf_roc = _axes[0].boxplot(_rf_roc_data, positions=_positions_rf, widths=0.4, patch_artist=False)
    _bp_xgb_roc = _axes[0].boxplot(_xgb_roc_data, positions=_positions_xgb, widths=0.4, patch_artist=False)

    _style_boxplots(_bp_knn_roc, "#eab308")  # Yellow for KNN
    _style_boxplots(_bp_lr_roc, "#10b981")   # Green for LR
    _style_boxplots(_bp_rf_roc, "#a78bfa")   # Purple for RF
    _style_boxplots(_bp_xgb_roc, "#3b82f6")  # Blue for XGBoost

    _axes[0].set_title("10-Fold CV ROC-AUC across Splits", fontsize=12, fontweight="bold", pad=15, color="#F1F5F9")
    _axes[0].set_xticks([2.5, 7.5, 12.5, 17.5])
    _axes[0].set_xticklabels(_split_labels, rotation=15)
    _axes[0].set_ylabel("ROC-AUC Score", fontsize=10)
    _axes[0].grid(True, linestyle="--", alpha=0.3)

    from matplotlib.lines import Line2D
    _legend_elements = [
        Line2D([0], [0], color='#eab308', lw=2.5, label='KNN'),
        Line2D([0], [0], color='#10b981', lw=2.5, label='Logistic Regression'),
        Line2D([0], [0], color='#a78bfa', lw=2.5, label='Random Forest'),
        Line2D([0], [0], color='#3b82f6', lw=2.5, label='XGBoost')
    ]
    _axes[0].legend(handles=_legend_elements, facecolor="#1e293b", edgecolor="#475569")

    # 2. PR-AUC Grouped Boxplot
    _knn_pr_data = [cv_results_knn[s]["pr_auc"] for s in _splits]
    _lr_pr_data = [cv_results_lr[s]["pr_auc"] for s in _splits]
    _rf_pr_data = [cv_results_rf[s]["pr_auc"] for s in _splits]
    _xgb_pr_data = [cv_results_xgb[s]["pr_auc"] for s in _splits]

    _bp_knn_pr = _axes[1].boxplot(_knn_pr_data, positions=_positions_knn, widths=0.4, patch_artist=False)
    _bp_lr_pr = _axes[1].boxplot(_lr_pr_data, positions=_positions_lr, widths=0.4, patch_artist=False)
    _bp_rf_pr = _axes[1].boxplot(_rf_pr_data, positions=_positions_rf, widths=0.4, patch_artist=False)
    _bp_xgb_pr = _axes[1].boxplot(_xgb_pr_data, positions=_positions_xgb, widths=0.4, patch_artist=False)

    _style_boxplots(_bp_knn_pr, "#f59e0b")  # Orange for KNN
    _style_boxplots(_bp_lr_pr, "#059669")   # Dark Green for LR
    _style_boxplots(_bp_rf_pr, "#ec4899")   # Pink for RF
    _style_boxplots(_bp_xgb_pr, "#06b6d4")  # Cyan for XGBoost

    _axes[1].set_title("10-Fold CV PR-AUC across Splits", fontsize=12, fontweight="bold", pad=15, color="#F1F5F9")
    _axes[1].set_xticks([2.5, 7.5, 12.5, 17.5])
    _axes[1].set_xticklabels(_split_labels, rotation=15)
    _axes[1].set_ylabel("PR-AUC Score", fontsize=10)
    _axes[1].grid(True, linestyle="--", alpha=0.3)

    _legend_elements_pr = [
        Line2D([0], [0], color='#f59e0b', lw=2.5, label='KNN'),
        Line2D([0], [0], color='#059669', lw=2.5, label='Logistic Regression'),
        Line2D([0], [0], color='#ec4899', lw=2.5, label='Random Forest'),
        Line2D([0], [0], color='#06b6d4', lw=2.5, label='XGBoost')
    ]
    _axes[1].legend(handles=_legend_elements_pr, facecolor="#1e293b", edgecolor="#475569")

    plt.tight_layout()
    _cv_plot = mo.as_html(_fig)
    plt.close(_fig)
    _cv_plot
    return


@app.cell
def _():
    mo.md(r"""
    ### 6.2 All Models Performance Summary
    """)
    return


@app.cell
def _(cv_results_knn, cv_results_lr, cv_results_rf, cv_results_xgb):
    _splits = ["warm_start_1_10", "warm_start_1_1", "protein_coldstart", "drug_coldstart"]
    _split_names = ["Warm-Start (1:10)", "Warm-Start (1:1)", "Protein Cold-Start", "Drug Cold-Start"]

    _rows = []
    for _s, _name in zip(_splits, _split_names):
        _knn_roc_mean = np.mean(cv_results_knn[_s]["roc_auc"])
        _knn_roc_std = np.std(cv_results_knn[_s]["roc_auc"])
        _knn_pr_mean = np.mean(cv_results_knn[_s]["pr_auc"])
        _knn_pr_std = np.std(cv_results_knn[_s]["pr_auc"])

        _lr_roc_mean = np.mean(cv_results_lr[_s]["roc_auc"])
        _lr_roc_std = np.std(cv_results_lr[_s]["roc_auc"])
        _lr_pr_mean = np.mean(cv_results_lr[_s]["pr_auc"])
        _lr_pr_std = np.std(cv_results_lr[_s]["pr_auc"])

        _rf_roc_mean = np.mean(cv_results_rf[_s]["roc_auc"])
        _rf_roc_std = np.std(cv_results_rf[_s]["roc_auc"])
        _rf_pr_mean = np.mean(cv_results_rf[_s]["pr_auc"])
        _rf_pr_std = np.std(cv_results_rf[_s]["pr_auc"])

        _xgb_roc_mean = np.mean(cv_results_xgb[_s]["roc_auc"])
        _xgb_roc_std = np.std(cv_results_xgb[_s]["roc_auc"])
        _xgb_pr_mean = np.mean(cv_results_xgb[_s]["pr_auc"])
        _xgb_pr_std = np.std(cv_results_xgb[_s]["pr_auc"])

        _rows.append({
            "Evaluation Split": _name,
            "KNN Mean ROC-AUC": f"{_knn_roc_mean:.4f} (± {_knn_roc_std:.4f})",
            "LR Mean ROC-AUC": f"{_lr_roc_mean:.4f} (± {_lr_roc_std:.4f})",
            "RF Mean ROC-AUC": f"{_rf_roc_mean:.4f} (± {_rf_roc_std:.4f})",
            "XGB Mean ROC-AUC": f"{_xgb_roc_mean:.4f} (± {_xgb_roc_std:.4f})",
            "KNN Mean PR-AUC": f"{_knn_pr_mean:.4f} (± {_knn_pr_std:.4f})",
            "LR Mean PR-AUC": f"{_lr_pr_mean:.4f} (± {_lr_pr_std:.4f})",
            "RF Mean PR-AUC": f"{_rf_pr_mean:.4f} (± {_rf_pr_std:.4f})",
            "XGB Mean PR-AUC": f"{_xgb_pr_mean:.4f} (± {_xgb_pr_std:.4f})"
        })

    _comparison_df = pd.DataFrame(_rows)
    _comparison_df
    return


@app.cell
def _():
    mo.md(r"""
    ### 6.3 10-Fold CV Grouped Results Interpretation & Statistical Insights

    Evaluating our ensembles across multiple split conditions and algorithms provides several fundamental insights:

    **1. Warm-Start Ratio Sensitivity (1:10 vs 1:1):**
    - Under the **Warm-Start 1:10** setting, tree-based models and KNN achieve excellent results, while linear Logistic Regression performs reasonably but slightly lower due to structural linearity constraints.
    - When switched to the balanced **Warm-Start 1:1** split, performance increases significantly across all models, with tree ensembles exceeding **0.97** PR-AUC and KNN/LR showing marked improvements.

    **2. The Generalisation Gradients (Warm vs. Cold Starts):**
    - **Protein Cold-Start**: Performance is remarkably robust for almost all models (with XGBoost and Random Forest leading). Sequence-level target protein CTD descriptors generalize successfully to entirely unseen protein families due to conserved global physicochemical patterns.
    - **Drug Cold-Start**: Performance collapses severely for all models, highlighting that localized 2D Morgan fingerprints are highly sensitive to scaffold novelty. Unseen drug structures produce out-of-distribution feature spaces where distance-based KNN and linear models struggle to extrapolate.

    **3. Model Capability Gradients (KNN vs. LR vs. RF vs. GBDTs):**
    - **XGBoost & Random Forest**: Consistently dominate across all splits in both ROC-AUC and PR-AUC. This confirms that tree-based ensembles are highly capable of capturing complex non-linear combinations of molecular bits and continuous sequence features.
    - **KNN**: Performs well on warm-start splits but shows sensitivity in cold-start configurations due to high-dimensional distance sparsity (curse of dimensionality on 1171 features).
    - **Logistic Regression**: Serves as a solid baseline but is limited by linear decision boundaries, confirming the necessity of complex tree structures or non-linear graph embeddings.
    """)
    return


if __name__ == "__main__":
    app.run()
