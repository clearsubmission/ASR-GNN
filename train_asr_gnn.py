import os
import json
import math
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from asr_dataset import load_kg_dataset
from asr_gnn_model import ASRGNN, ASRGNNConfig


DATASET_DIRS = {
    "FB15k237": "/home/amangel/datasets/FB15k-237",
    "WN18RR": "/home/amangel/datasets/WN18RR",
    "YAGO310": "/home/amangel/datasets/YAGO3-10",
}


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=DATASET_DIRS.keys())
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--emb_dim", type=int, default=200)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lambda_l0", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_negatives", type=int, default=10)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--add_inverse_edges", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_negative_tails(num_entities: int, batch_size: int, num_negatives: int, device: str):
    return torch.randint(0, num_entities, (batch_size, num_negatives), device=device)


def ranking_loss(pos_scores, neg_scores, margin=1.0):
    pos = pos_scores.unsqueeze(1).expand_as(neg_scores)
    target = torch.ones_like(neg_scores)
    return F.margin_ranking_loss(pos, neg_scores, target, margin=margin)


@torch.no_grad()
def evaluate_filtered(model, triples, data, device, eval_batch_size=128):
    model.eval()

    edge_index = data.train_edge_index.to(device)
    edge_type = data.train_edge_type.to(device)

    ranks = []

    for start in range(0, triples.size(0), eval_batch_size):
        batch = triples[start:start + eval_batch_size].to(device)
        heads = batch[:, 0]
        rels = batch[:, 1]
        tails = batch[:, 2]

        node_repr, edge_gate, _ = model.encode_graph(heads, rels, edge_index, edge_type)

        for i in range(batch.size(0)):
            h = heads[i].item()
            r = rels[i].item()
            t = tails[i].item()

            all_tails = torch.arange(data.num_entities, device=device)
            h_vec = torch.full((data.num_entities,), h, dtype=torch.long, device=device)
            r_vec = torch.full((data.num_entities,), r, dtype=torch.long, device=device)

            cand_triples = torch.stack([h_vec, r_vec, all_tails], dim=1)
            scores = model.score_triples(node_repr, cand_triples)

            # filtered setting
            for cand_t in range(data.num_entities):
                if cand_t != t and (h, r, cand_t) in data.all_true_triples:
                    scores[cand_t] = -1e9

            true_score = scores[t].item()
            rank = int((scores > true_score).sum().item()) + 1
            ranks.append(rank)

    ranks_t = torch.tensor(ranks, dtype=torch.float)
    mrr = torch.mean(1.0 / ranks_t).item()
    hits1 = torch.mean((ranks_t <= 1).float()).item()
    hits3 = torch.mean((ranks_t <= 3).float()).item()
    hits10 = torch.mean((ranks_t <= 10).float()).item()

    return {
        "mrr": mrr,
        "hits@1": hits1,
        "hits@3": hits3,
        "hits@10": hits10,
        "mean_rank": ranks_t.mean().item(),
    }


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_dir = DATASET_DIRS[args.dataset]
    data = load_kg_dataset(dataset_dir, add_inverse_edges=args.add_inverse_edges)

    cfg = ASRGNNConfig(
        num_entities=data.num_entities,
        num_relations=data.num_relations,
        emb_dim=args.emb_dim,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lambda_l0=args.lambda_l0,
        device=device,
    )

    model = ASRGNN(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(
        TensorDataset(data.train),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    edge_index = data.train_edge_index.to(device)
    edge_type = data.train_edge_type.to(device)

    best_valid_mrr = -1.0
    best_metrics = None

    run_config = {
        "dataset": args.dataset,
        "dataset_dir": dataset_dir,
        "num_entities": data.num_entities,
        "num_relations": data.num_relations,
        "emb_dim": args.emb_dim,
        "num_layers": args.num_layers,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "lambda_l0": args.lambda_l0,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "num_negatives": args.num_negatives,
        "eval_batch_size": args.eval_batch_size,
        "seed": args.seed,
        "device": device,
    }
    save_json(os.path.join(args.output_dir, "config.json"), run_config)

    print(json.dumps(run_config, indent=2))

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_task = 0.0
        total_sparse = 0.0
        total_rho = 0.0
        n_batches = 0

        for (batch,) in train_loader:
            batch = batch.to(device)

            heads = batch[:, 0]
            rels = batch[:, 1]
            true_tails = batch[:, 2]

            neg_tails = sample_negative_tails(
                num_entities=data.num_entities,
                batch_size=batch.size(0),
                num_negatives=args.num_negatives,
                device=device,
            )

            optimizer.zero_grad()

            node_repr, edge_gate, l0_term = model.encode_graph(heads, rels, edge_index, edge_type)

            pos_scores = model.score_triples(node_repr, batch)

            neg_heads = heads.unsqueeze(1).expand(-1, args.num_negatives).reshape(-1)
            neg_rels = rels.unsqueeze(1).expand(-1, args.num_negatives).reshape(-1)
            neg_triples = torch.stack([neg_heads, neg_rels, neg_tails.reshape(-1)], dim=1)

            neg_scores = model.score_triples(node_repr, neg_triples).view(batch.size(0), args.num_negatives)

            task_loss = ranking_loss(pos_scores, neg_scores, margin=1.0)
            sparse_loss = args.lambda_l0 * l0_term.mean()
            loss = task_loss + sparse_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_task += task_loss.item()
            total_sparse += sparse_loss.item()
