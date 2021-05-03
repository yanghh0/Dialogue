
import torch
import torch.nn.functional as F
import torch.nn as nn


def loss_function_1(recon_x, x, mu, logvar):
    BCE_loss = nn.BCELoss(reduction='sum')
    reconstruction_loss = BCE_loss(recon_x, x)
    KL_divergence = -0.5 * torch.sum(1 + logvar - torch.exp(logvar) - mu.pow(2))
    return reconstruction_loss + KL_divergence


def loss_function_2(recon_x, x, mu, logvar):
    MSE_loss = nn.MSELoss(size_average=False)
    reconstruction_loss = MSE_loss(recon_x, x)
    KL_divergence = -0.5 * torch.sum(1 + logvar - torch.exp(logvar) - mu.pow(2))
    return reconstruction_loss + KL_divergence


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(in_features=784, out_features=400)
        self.fc2_mean = nn.Linear(in_features=400, out_features=20)
        self.fc2_logvar = nn.Linear(in_features=400, out_features=20)
        self.fc3 = nn.Linear(in_features=20, out_features=400)
        self.fc4 = nn.Linear(in_features=400, out_features=784)

    def encode(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc2_mean(x)
        logvar = self.fc2_logvar(x)
        return mu, logvar

    def decode(self, z):
        z = F.relu(self.fc3(z))
        z = F.sigmoid(self.fc4(z))
        return z

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        z = torch.randn_like(std) * std + mu
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrization(mu, logvar)
        return self.decode(z), mu, logvar


if __name__ == "__main__":
    model = VAE()