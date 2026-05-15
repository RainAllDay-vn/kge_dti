import copy

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from metrics_utils import pr_auc, roc_auc


class PairFeatureDataset(Dataset):
    def __init__(
        self,
        head_ids: np.ndarray,
        tail_ids: np.ndarray,
        dense_features: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> None:
        self.head_ids = torch.as_tensor(head_ids, dtype=torch.long)
        self.tail_ids = torch.as_tensor(tail_ids, dtype=torch.long)
        self.dense_features = torch.as_tensor(dense_features, dtype=torch.float32)
        self.labels = None if labels is None else torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return self.head_ids.shape[0]

    def __getitem__(self, index: int):
        item = {
            "head": self.head_ids[index],
            "tail": self.tail_ids[index],
            "feats": self.dense_features[index],
        }
        if self.labels is not None:
            item["label"] = self.labels[index]
        return item


class TorchNFM(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_tails: int,
        sparse_embedding_dim: int,
        dense_feature_dim: int,
        hidden_units: tuple[int, ...] = (128, 128),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.head_embedding = nn.Embedding(num_heads, sparse_embedding_dim)
        self.tail_embedding = nn.Embedding(num_tails, sparse_embedding_dim)

        layers: list[nn.Module] = []
        input_dim = sparse_embedding_dim + dense_feature_dim
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.head_embedding.weight)
        nn.init.xavier_uniform_(self.tail_embedding.weight)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, head_ids: torch.Tensor, tail_ids: torch.Tensor, dense_features: torch.Tensor) -> torch.Tensor:
        sparse_embeddings = torch.stack(
            [self.head_embedding(head_ids), self.tail_embedding(tail_ids)],
            dim=1,
        )
        summed = sparse_embeddings.sum(dim=1)
        squared_sum = summed * summed
        sum_squared = (sparse_embeddings * sparse_embeddings).sum(dim=1)
        bi_interaction = 0.5 * (squared_sum - sum_squared)
        model_input = torch.cat([bi_interaction, dense_features], dim=1)
        return self.mlp(model_input).squeeze(1)


def encode_pair_ids(
    pairs: pd.DataFrame,
    head_encoder: dict[str, int],
    tail_encoder: dict[str, int],
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    head_ids = pairs["head"].map(head_encoder)
    tail_ids = pairs["tail"].map(tail_encoder)
    if head_ids.isna().any() or tail_ids.isna().any():
        raise ValueError(f"{context} pairs contain sparse IDs missing from the encoder.")
    return head_ids.to_numpy(dtype=np.int64), tail_ids.to_numpy(dtype=np.int64)


def train_nfm(
    train_pairs: pd.DataFrame,
    train_labels: np.ndarray,
    test_pairs: pd.DataFrame,
    test_labels: np.ndarray,
    train_features: np.ndarray,
    test_features: np.ndarray,
    head_encoder: dict[str, int],
    tail_encoder: dict[str, int],
    sparse_embedding_dim: int,
    epochs: int,
    batch_size: int,
    device: str,
    seed: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    hidden_units: tuple[int, ...],
    patience: int,
) -> tuple[pd.DataFrame, float, pd.DataFrame, float, np.ndarray]:
    torch.manual_seed(seed)
    train_head_ids, train_tail_ids = encode_pair_ids(train_pairs, head_encoder, tail_encoder, "Training")
    test_head_ids, test_tail_ids = encode_pair_ids(test_pairs, head_encoder, tail_encoder, "Test")

    train_dataset = PairFeatureDataset(train_head_ids, train_tail_ids, train_features, train_labels)
    test_dataset = PairFeatureDataset(test_head_ids, test_tail_ids, test_features)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = TorchNFM(
        num_heads=len(head_encoder),
        num_tails=len(tail_encoder),
        sparse_embedding_dim=sparse_embedding_dim,
        dense_feature_dim=train_features.shape[1],
        hidden_units=hidden_units,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0
    min_delta = 0.0001

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in train_loader:
            head = batch["head"].to(device)
            tail = batch["tail"].to(device)
            feats = batch["feats"].to(device)
            label = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(head, tail, feats)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * label.shape[0]
            total_count += label.shape[0]

        epoch_loss = total_loss / max(total_count, 1)
        print(f"NFM epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.6f}")
        if epoch_loss < best_loss - min_delta:
            best_loss = epoch_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if patience > 0 and bad_epochs >= patience:
                print(f"NFM early stopping at epoch {epoch + 1}; best loss: {best_loss:.6f}")
                break

    model.load_state_dict(best_state)
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(
                batch["head"].to(device),
                batch["tail"].to(device),
                batch["feats"].to(device),
            )
            predictions.append(torch.sigmoid(logits).detach().cpu().numpy())
    pred = np.concatenate(predictions).reshape(-1)
    roc_curve, roc_value = roc_auc(test_labels, pred)
    pr_curve, pr_value = pr_auc(test_labels, pred)
    return roc_curve, roc_value, pr_curve, pr_value, pred
