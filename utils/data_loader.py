import os
import torch


class KGDataset:

    def __init__(self, path):

        self.path = path

        train = self.read_triples(os.path.join(path, "train.txt"))
        valid = self.read_triples(os.path.join(path, "valid.txt"))
        test = self.read_triples(os.path.join(path, "test.txt"))

        self.train_triples = train
        self.valid_triples = valid
        self.test_triples = test

        entities = set()
        relations = set()

        for h, r, t in train + valid + test:
            entities.add(h)
            entities.add(t)
            relations.add(r)

        self.entity2id = {e: i for i, e in enumerate(sorted(entities))}
        self.relation2id = {r: i for i, r in enumerate(sorted(relations))}

        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.id2relation = {v: k for k, v in self.relation2id.items()}

        self.num_entities = len(self.entity2id)
        self.num_relations = len(self.relation2id)

        self.train = self.convert(self.train_triples)
        self.valid = self.convert(self.valid_triples)
        self.test = self.convert(self.test_triples)

    def read_triples(self, file):

        triples = []

        with open(file) as f:
            for line in f:
                h, r, t = line.strip().split()
                triples.append((h, r, t))

        return triples

    def convert(self, triples):

        return [
            (
                self.entity2id[h],
                self.relation2id[r],
                self.entity2id[t]
            )
            for h, r, t in triples
        ]

    def get_train_tensor(self):

        return torch.tensor(self.train, dtype=torch.long)

    def get_valid_tensor(self):

        return torch.tensor(self.valid, dtype=torch.long)

    def get_test_tensor(self):

        return torch.tensor(self.test, dtype=torch.long)
