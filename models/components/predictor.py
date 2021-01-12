import copy

import torch
from torch import nn

from .layers import MLP, BaselineLayer
from .transformer import TransformerEncoder, TransformerEncoderLayer
from .position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned, PositionEmbeddingLearned3D


# =======================================================================
# Dummy
# =======================================================================

class DummyHiddenPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_encoder = nn.Sequential(nn.Linear(5, 7),)
        self.ptp_encoder = nn.Sequential(nn.Linear(16, 7),)
        self.predictor = nn.Sequential(nn.Linear(7, 5))

    def combine_embeddings(self, embeddings):
        return torch.sum(embeddings, axis=1)

    def forward(self, embeddings, ptp):
        combined_embeddings = self.combine_embeddings(embeddings)
        encoded_embeddings = self.embedding_encoder(combined_embeddings)
        encoded_ptp = self.ptp_encoder(ptp.view(ptp.shape[0], -1))
        hidden = self.predictor(encoded_embeddings + encoded_ptp)
        return hidden


# =======================================================================
# Basic Transformer
# =======================================================================

class TransformerHiddenPredictor_no_latent(nn.Module):
    def __init__(self, embedding_size, hparams):
        super().__init__()

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size, nhead=hparams.hidden_predictor_nhead, dim_feedforward=hparams.hidden_predictor_dim_feedforward)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=hparams.hidden_predictor_layers)
        self.position_encoding = PositionEmbeddingLearned(embedding_size)
        self.query_embed = nn.Embedding(1, embedding_size)

        self.last_mlp = MLP(
            embedding_size, embedding_size, embedding_size, 3)

        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, x, ptp, latent):
        bs = len(x)

        token = self.query_embed.weight.unsqueeze(1).repeat(bs, 1, 1)
        x = torch.cat((x, token), dim=1)

        pos = self.position_encoding(x).permute(1, 0, 2)
        x = x.permute(1, 0, 2)
        x = x + pos
        x = self.transformer_encoder(x)

        x = x.permute(1, 0, 2)[:, -1]
        x = self.last_mlp(x)

        return x


class TransformerHiddenPredictor_no_latent_3D(nn.Module):
    def __init__(self, embedding_size, hparams):
        super().__init__()

        self.expand_num = 2 * 2

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size, nhead=hparams.hidden_predictor_nhead, dim_feedforward=hparams.hidden_predictor_dim_feedforward)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=hparams.hidden_predictor_layers)
        self.position_encoding = PositionEmbeddingLearned3D(embedding_size)
        self.query_embed = nn.Embedding(1, embedding_size * self.expand_num)

        self.conv1 = nn.Conv2d(embedding_size, embedding_size, 1, 1, 0)
        self.batchnorm1 = nn.BatchNorm2d(embedding_size)
        self.conv2 = nn.Conv2d(embedding_size, embedding_size, 1, 1, 0)
        self.batchnorm2 = nn.BatchNorm2d(embedding_size)
        self.relu = nn.ReLU()

        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, x, ptp, latent):
        bs, frame_num, h, w, feat_num = x.shape

        token = self.query_embed.weight.reshape(1, 2, 2, feat_num).unsqueeze(1).repeat(bs, 1, 1, 1, 1)

        x = torch.cat((x, token), dim=1)

        pos = self.position_encoding(x)
        x = x + pos
        x = x.reshape(bs, -1, feat_num).permute(1, 0, 2)
        x = self.transformer_encoder(x)

        x = x.permute(1, 0, 2).reshape(bs, frame_num+1, h, w, feat_num)[:, -1].permute(0, 3, 1, 2)
        x = self.batchnorm2(self.conv2(self.relu(self.batchnorm1(self.conv1(x)))))

        return x


# =======================================================================
# Transformer 1
# =======================================================================

class TransformerHiddenPredictor1(nn.Module):
    def __init__(self, embedding_size, hparams):
        super().__init__()

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size, nhead=hparams.hidden_predictor_nhead, dim_feedforward=hparams.hidden_predictor_dim_feedforward)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=hparams.hidden_predictor_layers)
        self.position_encoding = PositionEmbeddingLearned(embedding_size)
        self.query_embed = nn.Embedding(1, embedding_size)

        self.latent_expander = MLP(
            hparams.latent_size, embedding_size, embedding_size, 3)
        self.last_mlp = MLP(
            embedding_size, embedding_size, embedding_size, 3)

        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, x, ptp, latent):
        bs = len(x)

        latent = self.latent_expander(latent)

        x = torch.cat((x, self.query_embed.weight.unsqueeze(1).repeat(bs, 1, 1), latent.unsqueeze(1)), dim=1)

        pos = self.position_encoding(x).permute(1, 0, 2)
        x = x.permute(1, 0, 2)
        x = x + pos
        x = self.transformer_encoder(x)

        x = x.permute(1, 0, 2)[:, -2]
        x = self.last_mlp(x)

        return x


class TransformerHiddenPredictor1_ptp(nn.Module):
    def __init__(self, embedding_size, hparams):
        super().__init__()

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size, nhead=hparams.hidden_predictor_nhead, dim_feedforward=hparams.hidden_predictor_dim_feedforward)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=hparams.hidden_predictor_layers)
        self.position_encoding = PositionEmbeddingLearned(embedding_size)
        self.query_embed = nn.Embedding(1, embedding_size)

        self.ptp_expander = MLP(
            hparams.ptp_size, embedding_size, embedding_size, 3)
        self.last_mlp = MLP(
            embedding_size, embedding_size, embedding_size, 3)

        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, x, ptp, latent):
        bs = len(x)

        ptp = self.ptp_expander(ptp)

        x = torch.cat((x, self.query_embed.weight.unsqueeze(1).repeat(bs, 1, 1), ptp.unsqueeze(1)), dim=1)

        pos = self.position_encoding(x).permute(1, 0, 2)
        x = x.permute(1, 0, 2)
        x = x + pos
        x = self.transformer_encoder(x)

        x = x.permute(1, 0, 2)[:, -2]
        x = self.last_mlp(x)

        return x


class TransformerHiddenPredictor1_3D(nn.Module):
    def __init__(self, embedding_size, hparams):
        super().__init__()

        self.expand_num = 2 * 2

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size, nhead=hparams.hidden_predictor_nhead, dim_feedforward=hparams.hidden_predictor_dim_feedforward)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=hparams.hidden_predictor_layers)
        self.position_encoding = PositionEmbeddingLearned3D(embedding_size)
        self.query_embed = nn.Embedding(1, embedding_size * self.expand_num)

        self.latent_expander = MLP(
            hparams.latent_size, embedding_size, embedding_size * self.expand_num, 3)

        self.conv1 = nn.Conv2d(embedding_size, embedding_size, 1, 1, 0)
        self.batchnorm1 = nn.BatchNorm2d(embedding_size)
        self.conv2 = nn.Conv2d(embedding_size, embedding_size, 1, 1, 0)
        self.batchnorm2 = nn.BatchNorm2d(embedding_size)
        self.relu = nn.ReLU()

        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, x, ptp, latent):
        bs, frame_num, h, w, feat_num = x.shape

        latent = self.latent_expander(latent).reshape(bs, 2, 2, feat_num).unsqueeze(1)
        token = self.query_embed.weight.reshape(1, 2, 2, feat_num).unsqueeze(1).repeat(bs, 1, 1, 1, 1)

        x = torch.cat((x, token, latent), dim=1)

        pos = self.position_encoding(x)
        x = x + pos
        x = x.reshape(bs, -1, feat_num).permute(1, 0, 2)
        x = self.transformer_encoder(x)

        x = x.permute(1, 0, 2).reshape(bs, frame_num+2, h, w, feat_num)[:, -2].permute(0, 3, 1, 2)
        x = self.batchnorm2(self.conv2(self.relu(self.batchnorm1(self.conv1(x)))))

        return x


class TransformerHiddenPredictor1_ptp_3D(nn.Module):
    def __init__(self, embedding_size, hparams):
        super().__init__()

        self.expand_num = 2 * 2

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size, nhead=hparams.hidden_predictor_nhead, dim_feedforward=hparams.hidden_predictor_dim_feedforward)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=hparams.hidden_predictor_layers)
        self.position_encoding = PositionEmbeddingLearned3D(embedding_size)
        self.query_embed = nn.Embedding(1, embedding_size * self.expand_num)

        self.ptp_expander = MLP(
            hparams.ptp_size, embedding_size, embedding_size * self.expand_num, 3)

        self.conv1 = nn.Conv2d(embedding_size, embedding_size, 1, 1, 0)
        self.batchnorm1 = nn.BatchNorm2d(embedding_size)
        self.conv2 = nn.Conv2d(embedding_size, embedding_size, 1, 1, 0)
        self.batchnorm2 = nn.BatchNorm2d(embedding_size)
        self.relu = nn.ReLU()

        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, x, ptp, latent):
        bs, frame_num, h, w, feat_num = x.shape

        ptp = self.ptp_expander(ptp).reshape(bs, 2, 2, feat_num).unsqueeze(1)
        token = self.query_embed.weight.reshape(1, 2, 2, feat_num).unsqueeze(1).repeat(bs, 1, 1, 1, 1)

        x = torch.cat((x, token, ptp), dim=1)

        pos = self.position_encoding(x)
        x = x + pos
        x = x.reshape(bs, -1, feat_num).permute(1, 0, 2)
        x = self.transformer_encoder(x)

        x = x.permute(1, 0, 2).reshape(bs, frame_num+2, h, w, feat_num)[:, -2].permute(0, 3, 1, 2)
        x = self.batchnorm2(self.conv2(self.relu(self.batchnorm1(self.conv1(x)))))

        return x


# =======================================================================
# Transformer 2
# =======================================================================

class TransformerHiddenPredictor2(nn.Module):
    def __init__(self, embedding_size, hparams):
        super().__init__()

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size, nhead=hparams.hidden_predictor_nhead, dim_feedforward=hparams.hidden_predictor_dim_feedforward)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=hparams.hidden_predictor_layers)
        self.position_encoding = PositionEmbeddingLearned(embedding_size)
        self.query_embed = nn.Embedding(1, embedding_size)

        self.latent_expander = MLP(
            hparams.latent_size, embedding_size, embedding_size, 3)
        self.last_mlp = MLP(
            embedding_size, embedding_size, embedding_size, 3)

        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, x, ptp, latent):
        bs = len(x)

        latent = self.latent_expander(latent)
        token = self.query_embed.weight.unsqueeze(1).repeat(bs, 1, 1) + latent.unsqueeze(1)
        x = torch.cat((x, token), dim=1)

        pos = self.position_encoding(x).permute(1, 0, 2)
        x = x.permute(1, 0, 2)
        x = x + pos
        x = self.transformer_encoder(x)

        x = x.permute(1, 0, 2)[:, -1]
        x = self.last_mlp(x)

        return x


class TransformerHiddenPredictor2_ptp(nn.Module):
    def __init__(self, embedding_size, hparams):
        super().__init__()

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size, nhead=hparams.hidden_predictor_nhead, dim_feedforward=hparams.hidden_predictor_dim_feedforward)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=hparams.hidden_predictor_layers)
        self.position_encoding = PositionEmbeddingLearned(embedding_size)
        self.query_embed = nn.Embedding(1, embedding_size)

        self.ptp_expander = MLP(
            hparams.ptp_size, embedding_size, embedding_size, 3)
        self.last_mlp = MLP(
            embedding_size, embedding_size, embedding_size, 3)

        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, x, ptp, latent):
        bs = len(x)

        ptp = self.ptp_expander(ptp)
        token = self.query_embed.weight.unsqueeze(1).repeat(bs, 1, 1) + ptp.unsqueeze(1)
        x = torch.cat((x, token), dim=1)

        pos = self.position_encoding(x).permute(1, 0, 2)
        x = x.permute(1, 0, 2)
        x = x + pos
        x = self.transformer_encoder(x)

        x = x.permute(1, 0, 2)[:, -1]
        x = self.last_mlp(x)

        return x


class TransformerHiddenPredictor2_3D(nn.Module):
    def __init__(self, embedding_size, hparams):
        super().__init__()

        self.expand_num = 2 * 2

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size, nhead=hparams.hidden_predictor_nhead, dim_feedforward=hparams.hidden_predictor_dim_feedforward)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=hparams.hidden_predictor_layers)
        self.position_encoding = PositionEmbeddingLearned3D(embedding_size)
        self.query_embed = nn.Embedding(1, embedding_size * self.expand_num)

        self.latent_expander = MLP(
            hparams.latent_size, embedding_size, embedding_size * self.expand_num, 3)

        self.conv1 = nn.Conv2d(embedding_size, embedding_size, 1, 1, 0)
        self.batchnorm1 = nn.BatchNorm2d(embedding_size)
        self.conv2 = nn.Conv2d(embedding_size, embedding_size, 1, 1, 0)
        self.batchnorm2 = nn.BatchNorm2d(embedding_size)
        self.relu = nn.ReLU()

        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, x, ptp, latent):
        bs, frame_num, h, w, feat_num = x.shape

        latent = self.latent_expander(latent).reshape(bs, 2, 2, feat_num).unsqueeze(1)
        token = self.query_embed.weight.reshape(1, 2, 2, feat_num).unsqueeze(1).repeat(bs, 1, 1, 1, 1)

        token = token + latent
        x = torch.cat((x, token), dim=1)

        pos = self.position_encoding(x)
        x = x + pos
        x = x.reshape(bs, -1, feat_num).permute(1, 0, 2)
        x = self.transformer_encoder(x)

        x = x.permute(1, 0, 2).reshape(bs, frame_num+1, h, w, feat_num)[:, -1].permute(0, 3, 1, 2)
        x = self.batchnorm2(self.conv2(self.relu(self.batchnorm1(self.conv1(x)))))

        return x


class TransformerHiddenPredictor2_ptp_3D(nn.Module):
    def __init__(self, embedding_size, hparams):
        super().__init__()

        self.expand_num = 2 * 2

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size, nhead=hparams.hidden_predictor_nhead, dim_feedforward=hparams.hidden_predictor_dim_feedforward)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=hparams.hidden_predictor_layers)
        self.position_encoding = PositionEmbeddingLearned3D(embedding_size)
        self.query_embed = nn.Embedding(1, embedding_size * self.expand_num)

        self.ptp_expander = MLP(
            hparams.ptp_size, embedding_size, embedding_size * self.expand_num, 3)

        self.conv1 = nn.Conv2d(embedding_size, embedding_size, 1, 1, 0)
        self.batchnorm1 = nn.BatchNorm2d(embedding_size)
        self.conv2 = nn.Conv2d(embedding_size, embedding_size, 1, 1, 0)
        self.batchnorm2 = nn.BatchNorm2d(embedding_size)
        self.relu = nn.ReLU()

        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, x, ptp, latent):
        bs, frame_num, h, w, feat_num = x.shape

        ptp = self.ptp_expander(ptp).reshape(bs, 2, 2, feat_num).unsqueeze(1)
        token = self.query_embed.weight.reshape(1, 2, 2, feat_num).unsqueeze(1).repeat(bs, 1, 1, 1, 1)

        token = token + ptp
        x = torch.cat((x, token), dim=1)

        pos = self.position_encoding(x)
        x = x + pos
        x = x.reshape(bs, -1, feat_num).permute(1, 0, 2)
        x = self.transformer_encoder(x)

        x = x.permute(1, 0, 2).reshape(bs, frame_num+1, h, w, feat_num)[:, -1].permute(0, 3, 1, 2)
        x = self.batchnorm2(self.conv2(self.relu(self.batchnorm1(self.conv1(x)))))

        return x


# =======================================================================
# Transformer 3
# =======================================================================

class TransformerHiddenPredictor3(nn.Module):
    def __init__(self, embedding_size, hparams):
        super().__init__()

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size, nhead=hparams.hidden_predictor_nhead, dim_feedforward=hparams.hidden_predictor_dim_feedforward)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=hparams.hidden_predictor_layers)
        self.position_encoding = PositionEmbeddingLearned(embedding_size)

        self.latent_expander = MLP(
            hparams.latent_size, embedding_size, embedding_size, 3)
        self.last_mlp = MLP(
            embedding_size, embedding_size, embedding_size, 3)

        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, x, ptp, latent):
        bs = len(x)

        latent = self.latent_expander(latent)
        x = torch.cat((x, latent.unsqueeze(1)), dim=1)

        pos = self.position_encoding(x).permute(1, 0, 2)
        x = x.permute(1, 0, 2)
        x = x + pos
        x = self.transformer_encoder(x)

        x = x.permute(1, 0, 2)[:, -1]
        x = self.last_mlp(x)

        return x


# =======================================================================
# Addition predictor
# =======================================================================

class AddHiddenPredictor(nn.Module):
    def __init__(self, ptp_size, latent_size, num_conditional_frames):
        super().__init__()
        embedding_size = 64

        self.embedding_fc = MLP(embedding_size * 2, 256, embedding_size, 3)
        self.latent_fc = MLP(latent_size, 256, embedding_size, 3)
        self.combine_fc = MLP(embedding_size, 256, embedding_size, 3)

    def forward(self, x, ptp, latent):
        bs = x.shape[0]
        x = x.view(bs, -1)
        x = self.embedding_fc(x)
        latent = self.latent_fc(latent)
        x = x + latent
        x = self.combine_fc(x)

        return x


class AddHiddenPredictor_no_latent(nn.Module):
    def __init__(self, ptp_size, latent_size, num_conditional_frames):
        super().__init__()
        embedding_size = 64

        self.embedding_fc = MLP(embedding_size * 2, 256, embedding_size, 3)

    def forward(self, x, ptp, latent):
        bs = x.shape[0]
        x = x.view(bs, -1)
        x = self.embedding_fc(x)

        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# =======================================================================
# Baseline
# =======================================================================

class BaselineHiddenPredictor_no_latent(nn.Module):
    def __init__(self, embedding_size, hparams):
        super().__init__()

        input_dim = hparams.num_conditional_frames * embedding_size
        baseline_layer = BaselineLayer(
            input_dim,
            hparams.hidden_predictor_dim_feedforward
        )
        self.layers = _get_clones(
            baseline_layer, hparams.hidden_predictor_layers)

        self.last_mlp = MLP(
            input_dim, embedding_size, embedding_size, 3)

    def forward(self, x, ptp, latent):
        bs = x.shape[0]
        x = x.view(bs, -1)

        for layer in self.layers:
            x = layer(x)

        x = self.last_mlp(x)

        return x


class BaselineHiddenPredictor(nn.Module):
    def __init__(self, embedding_size, hparams):
        super().__init__()

        input_dim = (hparams.num_conditional_frames + 1) * embedding_size
        baseline_layer = BaselineLayer(
            input_dim,
            hparams.hidden_predictor_dim_feedforward
        )
        self.layers = _get_clones(
            baseline_layer, hparams.hidden_predictor_layers)

        self.last_mlp = MLP(
            input_dim, embedding_size, embedding_size, 3)

        self.latent_fc = MLP(hparams.latent_size, 256, embedding_size, 3)

    def forward(self, x, ptp, latent):
        bs = x.shape[0]
        x = x.view(bs, -1)

        latent = self.latent_fc(latent)
        x = torch.cat((x, latent), dim=1)

        for layer in self.layers:
            x = layer(x)

        x = self.last_mlp(x)

        return x


class BaselineHiddenPredictor_ptp(nn.Module):
    def __init__(self, embedding_size, hparams):
        super().__init__()

        input_dim = (hparams.num_conditional_frames + 1) * embedding_size
        baseline_layer = BaselineLayer(
            input_dim,
            hparams.hidden_predictor_dim_feedforward
        )
        self.layers = _get_clones(
            baseline_layer, hparams.hidden_predictor_layers)

        self.last_mlp = MLP(
            input_dim, embedding_size, embedding_size, 3)

        self.ptp_fc = MLP(hparams.ptp_size, 256, embedding_size, 3)

    def forward(self, x, ptp, latent):
        bs = x.shape[0]
        x = x.view(bs, -1)

        ptp = self.ptp_fc(ptp)
        x = torch.cat((x, ptp), dim=1)

        for layer in self.layers:
            x = layer(x)

        x = self.last_mlp(x)

        return x
