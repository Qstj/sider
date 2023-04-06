from torch_geometric.nn import GATConv, GCNConv, GraphConv, ResGatedGraphConv, global_max_pool
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, device, drug_features, heads=8, embed_dim=64, dropout=.125):
        super().__init__()
        self.device = device
        self.dropout = dropout
        self.drug_features = drug_features
        self.embed_dim = embed_dim

        side_effect_feature_paths = [
            'data/glove_wordEmbedding.pkl',
            'data/side_effect_label_750.pkl',
        ]
        side_effect_features = []
        for path in side_effect_feature_paths:
            with open(path, 'rb') as f:
                side_effect_features.append(pickle.load(f))
        self.side_effects = torch.from_numpy(np.concatenate(side_effect_features, axis=1)).float().to(device)

        self.x_encoder_1 = GATConv(drug_features[0], 96, heads=heads, dropout=dropout)
        self.x_encoder_2 = GATConv(96 * heads, 128, heads=heads, dropout=dropout)
        self.x_encoder_3 = GATConv(128 * heads, 128, heads=heads, dropout=dropout)
        self.x_encoder_4 = GATConv(128 * heads, 2 * embed_dim, dropout=dropout)

        self.x_encoder_5 = nn.Linear(2 * embed_dim, embed_dim)
        self.x_encoder_6 = nn.Linear(embed_dim, embed_dim)

        self.w_encoder_1 = nn.Linear(drug_features[1], drug_features[1])
        self.w_encoder_2 = nn.Linear(drug_features[1], embed_dim)

        self.z_encoder_1 = nn.Linear(drug_features[2], 1600)
        self.z_encoder_2 = nn.Linear(1600, embed_dim)

        self.xwz_aggregator = nn.Linear(embed_dim * 3, embed_dim)

        self.side_encoder = nn.Sequential(
            nn.Linear(self.side_effects.shape[1], embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh()
            )

    def drug_encoder(self, x):
        x, w, z = x

        x, edge_index, batch = x.x.to(self.device), x.edge_index.to(self.device), x.batch.to(self.device)
        x = F.relu_(self.x_encoder_1(x, edge_index))
        x = F.relu_(self.x_encoder_2(x, edge_index))
        x = F.relu_(self.x_encoder_3(x, edge_index))
        x = F.relu_(self.x_encoder_4(x, edge_index))
        x = global_max_pool(x, batch)
        x = F.relu_(self.x_encoder_5(x))
        x = F.dropout(x, p=.5, training=self.training)
        x = F.relu_(self.x_encoder_6(x))

        w = w.to(self.device)
        w = F.relu_(self.w_encoder_1(w))
        w = F.dropout(w, p=.5, training=self.training)
        w = F.relu_(self.w_encoder_2(w))

        z = z.to(self.device)
        z = F.relu_(self.z_encoder_1(z))
        z = F.dropout(z, p=.5, training=self.training)
        z = F.relu_(self.z_encoder_2(z))

        xwz = torch.tanh(torch.amax(torch.stack((x, w, z)), dim=0))

        return xwz

    def forward(self, x):
        drug_embedding = self.drug_encoder(x)
        side_embedding = self.side_encoder(self.side_effects.to(self.device))

        drug_vectors = F.normalize(drug_embedding, p=2, dim=1)
        side_vectors = F.normalize(side_embedding, p=2, dim=1)

        frequency = 5 * torch.matmul(drug_vectors, side_vectors.T)

        return frequency, drug_vectors, side_vectors
