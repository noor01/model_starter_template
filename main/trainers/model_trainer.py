import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import yaml
from argparse import Namespace

from networks.cell_state_networks import CellStateVAE
from dataloaders.cell_state_dataloader import (
    PreprocessedCellStateDataModule, collate_fn, 
    compute_reconstruction_metrics, compute_diversity_metrics,
    verify_preprocessed_data
)
from loggers.wandb_logger import WandbLogger


class CellStateTrainer:
    """Trainer for Cell State VAE with bidirectional training using preprocessed data."""
    
    def __init__(self, config, preprocessed_dir: str):
        """
        Args:
            config: Configuration object
            preprocessed_dir: Directory containing preprocessed data
        """
        self.config = config
        self.device = torch.device(config.device)
        self.preprocessed_dir = preprocessed_dir
        
        # Verify preprocessed data exists and get metadata
        print("Verifying preprocessed data...")
        self.data_info = verify_preprocessed_data(preprocessed_dir)
        self._print_data_info()
        
        # Initialize data module
        self.data_module = PreprocessedCellStateDataModule(preprocessed_dir, config)
        
        # Initialize model
        self.model = CellStateVAE(config).to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize logger
        self.logger = WandbLogger(config)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Create output directories
        os.makedirs(config.paths.checkpoint_dir, exist_ok=True)
        os.makedirs(config.paths.output_dir, exist_ok=True)
        
        # Mixed precision training
        self.use_amp = config.mixed_precision and config.device == "cuda"
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def _print_data_info(self):
        """Print information about the loaded preprocessed data."""
        info = self.data_info
        print(f"\nPreprocessed data info:")
        print(f"  Cells: {info['n_cells']:,}")
        print(f"  Features: {info['n_features']:,}")
        print(f"  K values: {info['k_values']}")
        print(f"  Training pairs: {info['n_training_pairs']:,}")
        print(f"  Data splits:")
        for split, size in info['data_splits'].items():
            print(f"    {split}: {size:,}")
        print(f"  Preprocessing config: {info['preprocessing_config']['distance_metric']} distance, "
              f"σ={info['preprocessing_config']['gaussian_sigma']}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        if self.config.optimizer.type.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.optimizer.weight_decay,
                betas=self.config.optimizer.betas
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer.type}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on config."""
        if self.config.scheduler.type.lower() == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.scheduler.T_max
            )
        elif self.config.scheduler.type.lower() == "step":
            return StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.config.scheduler.type.lower() == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler.type}")
    
    def compute_loss(self, model_output: Dict[str, torch.Tensor], 
                    beta1: float, beta2: float) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss with reconstruction and KL divergence terms.
        
        Args:
            model_output: Output from model forward pass
            beta1: Weight for global KL divergence
            beta2: Weight for local KL divergence
        
        Returns:
            Dictionary containing loss components
        """
        reconstructed = model_output['reconstructed']
        target = model_output['target']
        global_mu = model_output['global_mu']
        global_logvar = model_output['global_logvar']
        local_mu = model_output['local_mu']
        local_logvar = model_output['local_logvar']
        
        # Reconstruction loss
        if self.config.loss.reconstruction_loss == "mse":
            reconstruction_loss = F.mse_loss(reconstructed, target, reduction='mean')
        elif self.config.loss.reconstruction_loss == "l1":
            reconstruction_loss = F.l1_loss(reconstructed, target, reduction='mean')
        else:
            raise ValueError(f"Unknown reconstruction loss: {self.config.loss.reconstruction_loss}")
        
        # Global KL divergence
        global_kl_loss = -0.5 * torch.sum(
            1 + global_logvar - global_mu.pow(2) - global_logvar.exp()
        ) / global_mu.size(0)
        
        # Local KL divergence
        local_kl_loss = -0.5 * torch.sum(
            1 + local_logvar - local_mu.pow(2) - local_logvar.exp()
        ) / local_mu.size(0)
        
        # Total loss
        total_loss = reconstruction_loss + beta1 * global_kl_loss + beta2 * local_kl_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'global_kl_loss': global_kl_loss,
            'local_kl_loss': local_kl_loss
        }
    
    def get_kl_weights(self, epoch: int) -> Tuple[float, float]:
        """Get KL divergence weights with warmup."""
        warmup_epochs = self.config.loss.kl_warmup_epochs
        
        if epoch < warmup_epochs:
            # Linear warmup
            warmup_factor = epoch / warmup_epochs
            beta1 = self.config.training.beta1 * warmup_factor
            beta2 = self.config.training.beta2 * warmup_factor
        else:
            beta1 = self.config.training.beta1
            beta2 = self.config.training.beta2
        
        return beta1, beta2
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        # Move batch to device
        cell_embedding = batch['cell_embedding'].to(self.device)
        knn_embedding = batch['knn_embedding'].to(self.device)
        training_mode = batch['training_mode']
        
        # Get KL weights
        beta1, beta2 = self.get_kl_weights(self.current_epoch)
        
        # Forward pass
        if self.use_amp:
            with torch.cuda.amp.autocast():
                model_output = self.model(cell_embedding, knn_embedding, mode=training_mode)
                loss_dict = self.compute_loss(model_output, beta1, beta2)
            
            # Backward pass with mixed precision
            self.optimizer.zero_grad()
            self.scaler.scale(loss_dict['total_loss']).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            model_output = self.model(cell_embedding, knn_embedding, mode=training_mode)
            loss_dict = self.compute_loss(model_output, beta1, beta2)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            self.optimizer.step()
        
        # Convert losses to float
        metrics = {k: v.item() for k, v in loss_dict.items()}
        metrics['beta1'] = beta1
        metrics['beta2'] = beta2
        metrics['training_mode'] = training_mode
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """Validation loop."""
        self.model.eval()
        val_losses = []
        all_reconstructions = []
        all_targets = []
        
        val_loader = self.data_module.val_dataloader()
        
        with torch.no_grad():
            for batch in val_loader:
                cell_embedding = batch['cell_embedding'].to(self.device)
                knn_embedding = batch['knn_embedding'].to(self.device)
                training_mode = batch['training_mode']
                
                # Forward pass
                model_output = self.model(cell_embedding, knn_embedding, mode=training_mode)
                
                # Compute loss with current epoch's beta values
                beta1, beta2 = self.get_kl_weights(self.current_epoch)
                loss_dict = self.compute_loss(model_output, beta1, beta2)
                
                val_losses.append(loss_dict['total_loss'].item())
                all_reconstructions.append(model_output['reconstructed'].cpu())
                all_targets.append(model_output['target'].cpu())
        
        # Compute metrics
        avg_val_loss = np.mean(val_losses)
        
        # Compute reconstruction metrics on concatenated data
        all_reconstructions = torch.cat(all_reconstructions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        reconstruction_metrics = compute_reconstruction_metrics(all_reconstructions, all_targets)
        
        metrics = {
            'val_loss': avg_val_loss,
            **{f'val_{k}': v for k, v in reconstruction_metrics.items()}
        }
        
        return metrics
    
    def generate_and_evaluate_samples(self, n_samples: int = 50) -> Dict[str, float]:
        """Generate samples and compute diversity metrics."""
        self.model.eval()
        
        # Get a random validation sample as seed
        val_dataset = self.data_module.val_dataset
        seed_idx = np.random.randint(len(val_dataset))
        seed_sample = val_dataset[seed_idx]
        
        seed_embedding = seed_sample['cell_embedding'].unsqueeze(0).to(self.device)
        knn_embedding = seed_sample['knn_embedding'].unsqueeze(0).to(self.device)
        
        # Generate samples
        with torch.no_grad():
            generated_samples = self.model.generate_samples(
                seed_embedding, knn_embedding, n_samples=n_samples
            )
        
        # Compute diversity metrics
        diversity_metrics = compute_diversity_metrics(generated_samples.cpu())
        
        return {f'generation_{k}': v for k, v in diversity_metrics.items()}
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'preprocessed_dir': self.preprocessed_dir,
            'data_info': self.data_info
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config.paths.checkpoint_dir, 
            f'checkpoint_epoch_{epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(
                self.config.paths.checkpoint_dir, 
                'best_model.pt'
            )
            torch.save(checkpoint, best_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Check if checkpoint includes preprocessed data info
        if 'preprocessed_dir' in checkpoint:
            print(f"Checkpoint was trained with preprocessed data from: {checkpoint['preprocessed_dir']}")
        
        print(f"Checkpoint loaded: {checkpoint_path}")
    
    def train(self):
        """Main training loop."""
        print("Starting training...")
        
        # Log model architecture if enabled
        if self.config.logging.log_model_graph:
            sample_cell = torch.randn(1, self.config.model.embedding_size).to(self.device)
            sample_knn = torch.randn(1, self.config.model.embedding_size).to(self.device)
            self.logger.log_model_graph(
                "CellStateVAE", 
                self.config.paths.output_dir,
                (sample_cell, sample_knn),
                self.model
            )
        
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Set epoch for bidirectional training
            self.data_module.set_epoch(epoch)
            
            # Training loop
            train_loader = self.data_module.train_dataloader()
            epoch_metrics = []
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.training.num_epochs}")
            for batch_idx, batch in enumerate(pbar):
                metrics = self.train_step(batch)
                epoch_metrics.append(metrics)
                self.global_step += 1
                
                # Log metrics
                if self.global_step % self.config.logging.log_interval == 0:
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            self.logger.log_scalar(value, f"train/{key}", step=self.global_step)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{metrics['total_loss']:.4f}",
                    'recon': f"{metrics['reconstruction_loss']:.4f}",
                    'mode': metrics['training_mode']
                })
                
                # Validation
                if self.global_step % self.config.logging.eval_interval == 0:
                    val_metrics = self.validate()
                    
                    for key, value in val_metrics.items():
                        self.logger.log_scalar(value, key, step=self.global_step)
                    
                    # Generate samples for evaluation
                    if self.config.logging.log_images:
                        gen_metrics = self.generate_and_evaluate_samples()
                        for key, value in gen_metrics.items():
                            self.logger.log_scalar(value, key, step=self.global_step)
            
            # End of epoch validation
            val_metrics = self.validate()
            
            # Compute epoch averages
            epoch_avg = {}
            for key in epoch_metrics[0].keys():
                if isinstance(epoch_metrics[0][key], (int, float)):
                    epoch_avg[f"epoch_avg_{key}"] = np.mean([m[key] for m in epoch_metrics])
            
            # Log epoch metrics
            for key, value in {**val_metrics, **epoch_avg}.items():
                self.logger.log_scalar(value, key, step=epoch)
            
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
                self.logger.log_scalar(
                    self.optimizer.param_groups[0]['lr'], 
                    'learning_rate', 
                    step=epoch
                )
            
            # Save checkpoint
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
            
            if (epoch + 1) % self.config.logging.save_interval == 0:
                self.save_checkpoint(epoch, is_best=is_best)
            
            print(f"Epoch {epoch+1} - Train Loss: {epoch_avg['epoch_avg_total_loss']:.4f}, "
                  f"Val Loss: {val_metrics['val_loss']:.4f}")
        
        # Save final checkpoint
        self.save_checkpoint(self.config.training.num_epochs - 1)
        print("Training completed!")
    
    def close(self):
        """Clean up resources."""
        self.logger.close()


def load_config(config_path: str) -> Namespace:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    def dict_to_namespace(d):
        """Convert nested dictionary to nested namespace."""
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = dict_to_namespace(value)
        return Namespace(**d)
    
    return dict_to_namespace(config_dict)


def create_trainer(config_path: str, preprocessed_dir: str) -> CellStateTrainer:
    """Factory function to create trainer from preprocessed data."""
    config = load_config(config_path)
    return CellStateTrainer(config, preprocessed_dir)


# Main training script
def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Cell State VAE")
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to configuration file')
    parser.add_argument('--preprocessed_dir', type=str, required=True,
                       help='Path to preprocessed data directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Verify preprocessed data exists
    if not os.path.exists(args.preprocessed_dir):
        raise FileNotFoundError(f"Preprocessed data directory not found: {args.preprocessed_dir}")
    
    # Create trainer
    print(f"Loading preprocessed data from: {args.preprocessed_dir}")
    trainer = create_trainer(args.config, args.preprocessed_dir)
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Start training
    try:
        trainer.train()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
