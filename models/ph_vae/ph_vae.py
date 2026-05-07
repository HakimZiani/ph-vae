import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiDimPHVAE(nn.Module):
   

    def __init__(
        self,
        input_dim,
        latent_dim,
        n_phases=10,
        hidden_enc=(32, 16),
        hidden_dec=(16, 32),
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.m = n_phases

        self.encoder = self._build_mlp(input_dim, hidden_enc)
        self.fc_mu = nn.Linear(hidden_enc[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_enc[-1], latent_dim)

        self.decoder = self._build_mlp(latent_dim, hidden_dec)

        self.alpha_heads = nn.ModuleList([
            nn.Linear(hidden_dec[-1], n_phases) for _ in range(input_dim)
        ])
        self.lambda_heads = nn.ModuleList([
            nn.Linear(hidden_dec[-1], n_phases) for _ in range(input_dim)
        ])

        self._init_weights()

    def _build_mlp(self, input_size, hidden_sizes):
        layers = []
        current_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            current_size = hidden_size

        return nn.Sequential(*layers)

    

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


    def _build_A(self, lambdas):
        """
        Assemble the series-canonical (Coxian) sub-generator.

        lambdas : (B, m)  with  0 < lambda_1 <= ... <= lambda_m

        Returns
        -------
        A : (B, m, m)   upper-bidiagonal sub-generator
        t : (B, m)      exit-rate vector  t = -A @ 1
        """
        B, m = lambdas.shape
        device = lambdas.device

        # diagonal: -lambda_i
        # super-diagonal: +lambda_i  (state i transitions to i+1)
        idx = torch.arange(m, device=device)

        A = torch.zeros(B, m, m, device=device, dtype=lambdas.dtype)
        A[:, idx, idx]           = -lambdas                 
        A[:, idx[:-1], idx[1:]]  =  lambdas[:, :-1]         

        t = torch.zeros(B, m, device=device, dtype=lambdas.dtype)
        t[:, -1] = lambdas[:, -1]

        return A, t


    def encode(self, x):
        h      = self.encoder(x)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-10, 4)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)


    def decode_params(self, z):
        """
        Map latent z to PH parameters for every output dimension.

        Returns
        -------
        alphas : list of (B, m)    — initial distributions
        As     : list of (B, m, m) — sub-generators
        ts     : list of (B, m)    — exit-rate vectors
        """
        h = self.decoder(z)

        alphas, As, ts = [], [], []

        for j in range(self.input_dim):

            # initial distribution: proper simplex via softmax
            alpha_j = F.softmax(self.alpha_heads[j](h), dim=-1)

            # ordered positive rates:
            #   softplus keeps each increment > 0
            #   cumsum enforces lambda_1 <= lambda_2 <= ... <= lambda_m
            #   + small floor (0.01) prevents near-zero rates which make
            #     the uniformization parameter lambda very small and the
            #     series very long — a common source of NaNs
            increments = F.softplus(self.lambda_heads[j](h)) + 0.01
            lambdas    = torch.cumsum(increments, dim=-1)

            A_j, t_j = self._build_A(lambdas)

            alphas.append(alpha_j)
            As.append(A_j)
            ts.append(t_j)

        return alphas, As, ts


    def forward(self, x):
        """
        x : (B, input_dim)

        Returns
        -------
        mu, logvar : encoder output
        alphas, As, ts : PH parameters per dimension
        """
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        alphas, As, ts = self.decode_params(z)
        return mu, logvar, alphas, As, ts