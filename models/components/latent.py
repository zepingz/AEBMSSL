import torch
from torch import nn


class DummyLatentMerger(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.expander = nn.Sequential(nn.Linear(self.latent_size, 5))

    def forward(self, h, z):
        return self.expander(z) + h


class ResNetLatentMerger(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.expander = nn.Linear(self.latent_size, 512)

    def forward(self, h, z):
        return self.expander(z) + h


class TestLatentMerger(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.expander = nn.Linear(self.latent_size, 64)

    def forward(self, h, z):
        return self.expander(z) + h


class LinearizeLatentMerger1(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.expander = nn.Linear(self.latent_size - 1, 64)

    def forward(self, h, z):
        coef = z[:, :1]
        z = self.expander(z[:, 1:])
        delta_h = h[:, 1] - h[:, 0]
        new_h = h[:, 1] + (1 + coef) * delta_h + z
        return new_h


class LinearizeLatentMerger2(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.expander = nn.Linear(self.latent_size, 64)

    def forward(self, h, z):
        z = self.expander(z)
        delta_h = h[:, 1] - h[:, 0]
        new_h = h[:, 1] + torch.mul(h[:, 1] + delta_h, z)
        return new_h


class LinearizeLatentMerger3(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.expander = nn.Linear(self.latent_size, 64)

    def forward(self, h, z):
        z = self.expander(z)
        delta_h = h[:, 1] - h[:, 0]
        new_h = h[:, 1] + delta_h + z
        return new_h


class NewLinearizeLatentMerger(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.expander = nn.Linear(self.latent_size, 64)
        # self.fc = nn.Linear(64, 64)

    def forward(self, h, latent):
        latent = self.expander(latent)
        # h = self.fc(h)
        return torch.mul(h, latent)
        # return torch.mul(h, 1 + latent)
        # return h + latent

class NewLinearizeLatentMerger2(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.expander = nn.Linear(self.latent_size, 64)

    def forward(self, h, latent):
        latent = self.expander(latent)
        return torch.mul(h, 1 + latent)

class NewLinearizeLatentMerger3(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.expander = nn.Linear(self.latent_size, 64)

    def forward(self, h, latent):
        latent = self.expander(latent)
        return h + latent


###################################################################


class DummyLatentPredictor(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.model = nn.Sequential(nn.Linear(5, self.latent_size))

    def forward(self, target_embedding, conditional_embedding):
        return self.model(target_embedding + conditional_embedding)


class ResNetLatentPredictor(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.fc = nn.Linear(512, self.latent_size)

    def forward(self, target_embedding, conditional_embedding):
        return self.fc(target_embedding + conditional_embedding)


class TestLatentPredictor(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.fc = nn.Linear(64, self.latent_size)

    def forward(self, target_embedding, conditional_embedding):
        return self.fc(target_embedding - conditional_embedding)


class LinearizeLatentPredictor1(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.project = nn.Linear(64, self.latent_size)
        self.project_embedding = nn.Linear(64 * 2, 64)

    def forward(self, target_embedding, conditional_embedding):
        bs = conditional_embedding.shape[0]
        conditional_embedding = conditional_embedding.view(bs, -1)
        conditional_embedding = self.project_embedding(conditional_embedding)
        return self.project(target_embedding - conditional_embedding)


class LinearizeLatentPredictor2(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.project = nn.Linear(64 * 3, self.latent_size)

    def forward(self, target_embedding, conditional_embedding):
        combined_embedding = conditional_embedding.view(
            conditional_embedding.shape[0], -1
        )
        combined_embedding = torch.cat(
            (combined_embedding, target_embedding), dim=-1
        )
        return self.project(combined_embedding)


class LinearizeLatentPredictor3(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.project = nn.Linear(64, self.latent_size)
        self.project_embedding = nn.Linear(64 * 2, 64)

    def forward(self, target_embedding, conditional_embedding):
        bs = conditional_embedding.shape[0]
        conditional_embedding = conditional_embedding.view(bs, -1)
        conditional_embedding = self.project_embedding(conditional_embedding)
        guessed_embedding = (
            2 * conditional_embedding[:, 1] - conditional_embedding[:, 0]
        )
        return self.project(target_embedding - guessed_embedding)


class NewLinearizeLatentPredictor(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.project = nn.Linear(64, self.latent_size)

    def forward(self, target_embedding, predicted_hidden):
        bs = predicted_hidden.shape[0]
        predicted_hidden = predicted_hidden.view(bs, -1)
        return self.project(target_embedding - predicted_hidden)
