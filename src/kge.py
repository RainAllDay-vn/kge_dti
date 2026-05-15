from typing import Iterable

import numpy as np
import pandas as pd
import torch

from constants import TRIPLE_COLUMNS


def require_entities(factory, entities: Iterable[str], context: str) -> None:
    missing = sorted(set(entities).difference(factory.entity_to_id))
    if missing:
        examples = ", ".join(missing[:10])
        raise ValueError(
            f"{context} has {len(missing)} entities missing from the PyKEEN training graph. "
            f"Examples: {examples}"
        )


def train_distmult(
    train_triples: pd.DataFrame,
    device: str,
    embedding_dim: int,
    epochs: int,
    batch_size: int,
    seed: int,
    lr: float,
    num_negs_per_pos: int,
    use_tqdm: bool,
):
    from pykeen.models import DistMult
    from pykeen.training import SLCWATrainingLoop
    from pykeen.triples import TriplesFactory

    triples_factory = TriplesFactory.from_labeled_triples(train_triples[TRIPLE_COLUMNS].to_numpy(dtype=str))
    model = DistMult(
        triples_factory=triples_factory,
        embedding_dim=embedding_dim,
        loss="MarginRankingLoss",
        loss_kwargs={"margin": 1.0},
        random_seed=seed,
    ).to(device)
    training_loop = SLCWATrainingLoop(
        model=model,
        triples_factory=triples_factory,
        optimizer="adam",
        optimizer_kwargs={"lr": lr},
        negative_sampler="basic",
        negative_sampler_kwargs={"num_negs_per_pos": num_negs_per_pos},
    )
    losses = training_loop.train(
        triples_factory=triples_factory,
        num_epochs=epochs,
        batch_size=batch_size,
        use_tqdm=use_tqdm,
        use_tqdm_batch=use_tqdm,
    )
    return model, triples_factory, losses


def entity_embeddings(model, triples_factory, labels: Iterable[str], device: str) -> np.ndarray:
    ids = triples_factory.entities_to_ids(list(labels))
    indices = torch.as_tensor(ids, dtype=torch.long, device=device)
    with torch.no_grad():
        representation = model.entity_representations[0](indices=indices)
    return representation.detach().cpu().numpy()


def pair_embeddings(model, triples_factory, pairs: pd.DataFrame, device: str) -> np.ndarray:
    heads = entity_embeddings(model, triples_factory, pairs["head"].astype(str).tolist(), device)
    tails = entity_embeddings(model, triples_factory, pairs["tail"].astype(str).tolist(), device)
    return np.concatenate([heads, tails], axis=1)


def score_triples(model, triples_factory, triples: pd.DataFrame, device: str, batch_size: int) -> np.ndarray:
    mapped = triples_factory.map_triples(triples[TRIPLE_COLUMNS].to_numpy(dtype=str))
    scores = []
    model.eval()
    with torch.no_grad():
        for start in range(0, mapped.shape[0], batch_size):
            batch = mapped[start : start + batch_size].to(device)
            score = model.predict_hrt(batch).detach().cpu().reshape(-1).numpy()
            scores.append(score)
    return np.concatenate(scores)
