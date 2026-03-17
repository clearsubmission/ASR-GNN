import torch
import torch.nn as nn
import torch.nn.functional as F


class ASRGNN(nn.Module):
    def __init__(self, num_entities, num_relations, emb_dim=200, top_k=5):
        super().__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.emb_dim = emb_dim
        self.top_k = top_k

        # Embeddings
        self.entity_emb = nn.Embedding(num_entities, emb_dim)
        self.relation_emb = nn.Embedding(num_relations, emb_dim)

        # Relation selector (learn importance)
        self.rel_selector = nn.Linear(emb_dim, 1)

        # Message passing transformation
        self.linear = nn.Linear(emb_dim, emb_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, h, r, t, edge_index=None, edge_type=None):

        h_emb = self.entity_emb(h)
        r_emb = self.relation_emb(r)
        t_emb = self.entity_emb(t)

        # -------- Relation selection (core idea) --------
        rel_scores = self.rel_selector(self.relation_emb.weight).squeeze()

        topk = torch.topk(rel_scores, self.top_k).indices

        mask = torch.zeros_like(rel_scores)
        mask[topk] = 1.0

        r_masked = r_emb * mask[r].unsqueeze(-1)

        # -------- Simple message passing --------
        if edge_index is not None:

            src, dst = edge_index

            agg = torch.zeros_like(self.entity_emb.weight)

            agg.index_add_(0, dst, self.entity_emb(src))

            node_emb = self.entity_emb.weight + self.linear(agg)

        else:
            node_emb = self.entity_emb.weight

        # -------- Score function --------
        score = torch.sum((h_emb + r_masked) * t_emb, dim=1)

        return score

    def predict_all_tails(self, h, r):

        h_emb = self.entity_emb(h)
        r_emb = self.relation_emb(r)

        all_entities = self.entity_emb.weight

        scores = torch.matmul(h_emb + r_emb, all_entities.t())

        return scores
