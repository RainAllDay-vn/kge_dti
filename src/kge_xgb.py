"""CLI runner for the PyTorch KGE_XGBoost experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

from constants import TRIPLE_COLUMNS
from data_utils import (
    DatasetSpec,
    encode_labels,
    get_dataset_spec,
    load_all_dti,
    load_feature_tables,
    load_fold,
    load_kg,
    merge_features,
)
from kge import pair_embeddings, require_entities, score_triples, train_distmult
from metrics_utils import pr_auc, roc_auc
from utils import ensure_output_dirs, resolve_device, set_seed


def run_fold(
    fold: int,
    args: argparse.Namespace,
    spec: DatasetSpec,
    kg: pd.DataFrame,
    drug_df: pd.DataFrame,
    protein_df: pd.DataFrame,
    device: str,
) -> dict[str, float]:
    print(f"\n--- Fold {fold} ---")
    train, test = load_fold(spec, fold)
    train_pos = train.loc[train["label"] == 1, TRIPLE_COLUMNS]
    kge_train = pd.concat([train_pos, kg], ignore_index=True)[TRIPLE_COLUMNS].astype(str)

    # 1. Train DistMult KGE model
    kge_model, triples_factory, losses = train_distmult(
        kge_train,
        device=device,
        embedding_dim=args.embedding_dim,
        epochs=args.kge_epochs,
        batch_size=args.kge_batch_size,
        seed=args.seed + fold,
        lr=args.kge_lr,
        num_negs_per_pos=args.kge_num_negs,
        use_tqdm=not args.no_tqdm,
    )

    require_entities(triples_factory, train["head"].tolist() + train["tail"].tolist(), "Training fold")
    require_entities(triples_factory, test["head"].tolist() + test["tail"].tolist(), "Test fold")

    output_root = Path(args.output_dir)
    torch.save(
        {
            "model_state_dict": kge_model.state_dict(),
            "entity_to_id": triples_factory.entity_to_id,
            "relation_to_id": triples_factory.relation_to_id,
            "losses": losses,
            "args": vars(args),
        },
        output_root / "model" / f"kge_fold_{fold}.pt",
    )

    # 2. Evaluate KGE alone (DistMult score)
    test_labels = test["label"].to_numpy(dtype=np.float32)
    kge_scores = score_triples(kge_model, triples_factory, test[TRIPLE_COLUMNS], device, args.kge_batch_size)
    roc_curve, roc_value = roc_auc(test_labels, kge_scores)
    pr_curve, pr_value = pr_auc(test_labels, kge_scores)

    # 3. Extract features
    train_pairs = train[TRIPLE_COLUMNS]
    test_pairs = test[TRIPLE_COLUMNS]

    # KGE features
    train_kge_features = pair_embeddings(kge_model, triples_factory, train_pairs, device)
    test_kge_features = pair_embeddings(kge_model, triples_factory, test_pairs, device)

    # Descriptor features
    train_des = merge_features(train_pairs, drug_df, protein_df, spec, use_protein_features=not args.drug_features_only)
    test_des = merge_features(test_pairs, drug_df, protein_df, spec, use_protein_features=not args.drug_features_only)

    # 4. XGBoost Baseline (Structural descriptors only)
    print("Training XGBoost Baseline (Descriptors only)...")
    xgb_baseline_clf = XGBClassifier(
        n_estimators=args.xgb_estimators,
        learning_rate=args.xgb_lr,
        max_depth=args.xgb_max_depth,
        scale_pos_weight=args.xgb_scale_pos_weight,
        tree_method="hist",
        random_state=args.seed + fold,
        n_jobs=-1
    )
    # Scale descriptors
    scaler_des = MinMaxScaler(feature_range=(0, 1))
    train_des_scaled = scaler_des.fit_transform(train_des).astype(np.float32)
    test_des_scaled = scaler_des.transform(test_des).astype(np.float32)

    xgb_baseline_clf.fit(train_des_scaled, train["label"].to_numpy(dtype=np.float32))
    xgb_baseline_preds = xgb_baseline_clf.predict_proba(test_des_scaled)[:, 1]

    roc_xgb, roc_xgb_value = roc_auc(test_labels, xgb_baseline_preds)
    pr_xgb, pr_xgb_value = pr_auc(test_labels, xgb_baseline_preds)

    # 5. KGE + XGBoost (Concatenated features)
    print("Training KGE + XGBoost Classifier...")
    train_all_features = np.concatenate([train_kge_features, train_des], axis=1)
    test_all_features = np.concatenate([test_kge_features, test_des], axis=1)

    scaler_all = MinMaxScaler(feature_range=(0, 1))
    train_all_features = scaler_all.fit_transform(train_all_features).astype(np.float32)
    test_all_features = scaler_all.transform(test_all_features).astype(np.float32)

    kge_xgb_clf = XGBClassifier(
        n_estimators=args.xgb_estimators,
        learning_rate=args.xgb_lr,
        max_depth=args.xgb_max_depth,
        scale_pos_weight=args.xgb_scale_pos_weight,
        tree_method="hist",
        random_state=args.seed + fold,
        n_jobs=-1
    )
    kge_xgb_clf.fit(train_all_features, train["label"].to_numpy(dtype=np.float32))
    kge_xgb_preds = kge_xgb_clf.predict_proba(test_all_features)[:, 1]

    roc_kge_xgb, roc_kge_xgb_value = roc_auc(test_labels, kge_xgb_preds)
    pr_kge_xgb, pr_kge_xgb_value = pr_auc(test_labels, kge_xgb_preds)

    # Save curves
    roc_curve.to_csv(output_root / "curve" / "roc" / f"{fold}.csv", index=False)
    pr_curve.to_csv(output_root / "curve" / "pr" / f"{fold}.csv", index=False)
    roc_xgb.to_csv(output_root / "curve" / "roc_xgb" / f"{fold}.csv", index=False)
    pr_xgb.to_csv(output_root / "curve" / "pr_xgb" / f"{fold}.csv", index=False)
    roc_kge_xgb.to_csv(output_root / "curve" / "roc_kge_xgb" / f"{fold}.csv", index=False)
    pr_kge_xgb.to_csv(output_root / "curve" / "pr_kge_xgb" / f"{fold}.csv", index=False)

    # Save detailed predictions
    predictions = test[TRIPLE_COLUMNS + ["label"]].copy()
    predictions["kge_score"] = kge_scores
    predictions["xgb_baseline_pred"] = xgb_baseline_preds
    predictions["kge_xgb_pred"] = kge_xgb_preds
    predictions.to_csv(output_root / "predictions" / f"fold_{fold}.csv", index=False)

    print(f"KGE alone         - roc_auc: {roc_value:.6f}, pr_auc: {pr_value:.6f}")
    print(f"XGB Baseline      - roc_auc: {roc_xgb_value:.6f}, pr_auc: {pr_xgb_value:.6f}")
    print(f"KGE + XGB Classifier - roc_auc: {roc_kge_xgb_value:.6f}, pr_auc: {pr_kge_xgb_value:.6f}")

    return {
        "roc_auc_kge": roc_value,
        "pr_auc_kge": pr_value,
        "roc_auc_xgb": roc_xgb_value,
        "pr_auc_xgb": pr_xgb_value,
        "roc_auc_kge_xgb": roc_kge_xgb_value,
        "pr_auc_kge_xgb": pr_kge_xgb_value,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KGE_XGBoost with PyKEEN DistMult and XGBClassifier.")
    parser.add_argument("--dataset", default="yamanishi_08", help="Dataset name: yamanishi_08, BioKG, hetionet.")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Directory containing dataset folders. Defaults to <repo>/data.",
    )
    parser.add_argument("--split", default="warm_start_1_10", help="Fold split directory name.")
    parser.add_argument("--folds", type=int, default=10, help="Number of folds to run from fold 0.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--kge-epochs", type=int, default=50)
    parser.add_argument("--kge-batch-size", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=400, help="PyKEEN DistMult entity dimension.")
    parser.add_argument("--protein-pca-components", type=int, default=100)
    parser.add_argument("--kge-lr", type=float, default=1e-3)
    parser.add_argument("--kge-num-negs", type=int, default=1)
    
    # XGBoost Hyperparameters
    parser.add_argument("--xgb-estimators", type=int, default=500)
    parser.add_argument("--xgb-lr", type=float, default=0.05)
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-scale-pos-weight", type=float, default=10.0)

    parser.add_argument("--output-dir", default="output/kge_xgb_torch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--drug-features-only", action="store_true")
    parser.add_argument("--no-tqdm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    data_root = Path(args.data_root) if args.data_root is not None else repo_root / "data"
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_root = Path(args.output_dir)

    # Re-use ensure_output_dirs helper
    ensure_output_dirs(output_root)
    # Extra output dirs for the specific curves
    for curve in ["roc_xgb", "pr_xgb", "roc_kge_xgb", "pr_kge_xgb"]:
        (output_root / "curve" / curve).mkdir(parents=True, exist_ok=True)

    spec = get_dataset_spec(data_root, args.dataset, args.split)
    drug_df, protein_df = load_feature_tables(spec, pca_components=args.protein_pca_components)
    kg = load_kg(spec)

    metrics_by_fold = []
    for fold in range(args.folds):
        metrics_by_fold.append(
            run_fold(
                fold=fold,
                args=args,
                spec=spec,
                kg=kg,
                drug_df=drug_df,
                protein_df=protein_df,
                device=device,
            )
        )

    stable_metrics = pd.DataFrame(metrics_by_fold)
    print("\n=== Final Cross-Validation Summary ===")
    print(stable_metrics)
    print("\nSummary Statistics:")
    print(stable_metrics.describe())
    
    (output_root / "auc").mkdir(parents=True, exist_ok=True)
    stable_metrics.to_csv(output_root / "auc" / "kge_xgb_torch_auc.csv", index=False)


if __name__ == "__main__":
    main()
