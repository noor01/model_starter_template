import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List


class GlobalEncoder(nn.Module):
    """Global encoder that maps individual cell embeddings to shared latent space."""
    
    def __init__(self, embedding_size: int, latent_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        layers = []
        input_dim = embedding_size
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Separate heads for mean and log variance
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Cell embeddings [batch_size, embedding_size]
        Returns:
            mu: Mean of global latent distribution [batch_size, latent_dim]
            logvar: Log variance of global latent distribution [batch_size, latent_dim]
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class LocalEncoder(nn.Module):
    """Local encoder that maps KNN-averaged embeddings to neighborhood-specific latent variables."""
    
    def __init__(self, embedding_size: int, latent_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        layers = []
        input_dim = embedding_size
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Separate heads for mean and log variance
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: KNN-averaged embeddings [batch_size, embedding_size]
        Returns:
            mu: Mean of local latent distribution [batch_size, latent_dim]
            logvar: Log variance of local latent distribution [batch_size, latent_dim]
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """Decoder that reconstructs individual cells from concatenated global + local latent codes."""
    
    def __init__(self, global_latent_dim: int, local_latent_dim: int, 
                 embedding_size: int, hidden_dims: List[int]):
        super().__init__()
        
        # Reverse the hidden dimensions for decoder
        decoder_hidden_dims = hidden_dims[::-1]
        
        layers = []
        input_dim = global_latent_dim + local_latent_dim
        
        # Build hidden layers
        for hidden_dim in decoder_hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(input_dim, embedding_size))
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, global_z: torch.Tensor, local_z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_z: Global latent codes [batch_size, global_latent_dim]
            local_z: Local latent codes [batch_size, local_latent_dim]
        Returns:
            reconstructed: Reconstructed cell embeddings [batch_size, embedding_size]
        """
        # Concatenate global and local latent codes
        z = torch.cat([global_z, local_z], dim=1)
        reconstructed = self.decoder(z)
        return reconstructed


class CellStateVAE(nn.Module):
    """
    Hierarchical Variational Autoencoder for cell state modeling.
    
    Architecture:
    - Global Encoder: Cell → Global Latent (shared characteristics)
    - Local Encoder: KNN Average → Local Latent (neighborhood-specific)
    - Decoder: [Global + Local] → Reconstructed Cell
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.embedding_size = config.model.embedding_size
        self.global_latent_dim = config.model.global_latent_dim
        self.local_latent_dim = config.model.local_latent_dim
        self.hidden_dims = config.model.hidden_dims
        
        # Initialize encoders and decoder
        self.global_encoder = GlobalEncoder(
            self.embedding_size, 
            self.global_latent_dim, 
            self.hidden_dims
        )
        
        self.local_encoder = LocalEncoder(
            self.embedding_size, 
            self.local_latent_dim, 
            self.hidden_dims
        )
        
        self.decoder = Decoder(
            self.global_latent_dim,
            self.local_latent_dim,
            self.embedding_size,
            self.hidden_dims
        )
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, cell_embedding: torch.Tensor, 
                knn_embedding: torch.Tensor, 
                mode: str = "forward") -> Dict[str, torch.Tensor]:
        """
        Forward pass of the VAE.
        
        Args:
            cell_embedding: Individual cell embeddings [batch_size, embedding_size]
            knn_embedding: KNN-averaged embeddings [batch_size, embedding_size]
            mode: "forward" (KNN→Cell) or "reverse" (Cell→KNN)
        
        Returns:
            Dictionary containing:
                - reconstructed: Reconstructed embeddings
                - global_mu, global_logvar: Global latent parameters
                - local_mu, local_logvar: Local latent parameters
                - global_z, local_z: Sampled latent codes
        """
        
        if mode == "forward":
            # Forward mode: KNN Average → Individual Cell
            # Global encoding from individual cell
            global_mu, global_logvar = self.global_encoder(cell_embedding)
            
            # Local encoding from KNN average
            local_mu, local_logvar = self.local_encoder(knn_embedding)
            
            # Sample latent codes
            global_z = self.reparameterize(global_mu, global_logvar)
            local_z = self.reparameterize(local_mu, local_logvar)
            
            # Decode to reconstruct individual cell
            reconstructed = self.decoder(global_z, local_z)
            target = cell_embedding
            
        elif mode == "reverse":
            # Reverse mode: Individual Cell → KNN Average (regularization)
            # Global encoding from individual cell
            global_mu, global_logvar = self.global_encoder(cell_embedding)
            
            # Local encoding from individual cell (using cell as neighborhood proxy)
            local_mu, local_logvar = self.local_encoder(cell_embedding)
            
            # Sample latent codes
            global_z = self.reparameterize(global_mu, global_logvar)
            local_z = self.reparameterize(local_mu, local_logvar)
            
            # Decode to reconstruct KNN average
            reconstructed = self.decoder(global_z, local_z)
            target = knn_embedding
            
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'forward' or 'reverse'.")
        
        return {
            'reconstructed': reconstructed,
            'target': target,
            'global_mu': global_mu,
            'global_logvar': global_logvar,
            'local_mu': local_mu,
            'local_logvar': local_logvar,
            'global_z': global_z,
            'local_z': local_z
        }
    
    def generate_samples(self, seed_embedding: torch.Tensor, 
                        knn_embedding: torch.Tensor, 
                        n_samples: int = 100) -> torch.Tensor:
        """
        Generate diverse cell samples given a seed cell and its neighborhood.
        
        Args:
            seed_embedding: Seed cell embedding [1, embedding_size]
            knn_embedding: KNN-averaged embedding [1, embedding_size]
            n_samples: Number of samples to generate
        
        Returns:
            Generated cell embeddings [n_samples, embedding_size]
        """
        self.eval()
        with torch.no_grad():
            # Get latent distributions
            global_mu, global_logvar = self.global_encoder(seed_embedding)
            local_mu, local_logvar = self.local_encoder(knn_embedding)
            
            # Sample multiple times from the distributions
            samples = []
            for _ in range(n_samples):
                global_z = self.reparameterize(global_mu, global_logvar)
                local_z = self.reparameterize(local_mu, local_logvar)
                sample = self.decoder(global_z, local_z)
                samples.append(sample)
            
            return torch.cat(samples, dim=0)
    
    def encode(self, cell_embedding: torch.Tensor, 
               knn_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode cell and neighborhood to latent space."""
        global_mu, global_logvar = self.global_encoder(cell_embedding)
        local_mu, local_logvar = self.local_encoder(knn_embedding)
        
        global_z = self.reparameterize(global_mu, global_logvar)
        local_z = self.reparameterize(local_mu, local_logvar)
        
        return {
            'global_mu': global_mu,
            'global_logvar': global_logvar,
            'local_mu': local_mu,
            'local_logvar': local_logvar,
            'global_z': global_z,
            'local_z': local_z
        }
    
    def decode(self, global_z: torch.Tensor, local_z: torch.Tensor) -> torch.Tensor:
        """Decode latent codes to cell embeddings."""
        return self.decoder(global_z, local_z)
