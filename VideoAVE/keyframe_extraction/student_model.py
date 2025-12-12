"""
Student model for frame importance prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Any
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class FrameImportancePredictor(nn.Module):
    """
    Lightweight MLP for predicting frame importance scores
    """
    def __init__(self, vision_dim: int = 1024, hidden_dim: int = 512):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.hidden_dim = hidden_dim
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(vision_dim + 1, hidden_dim),  # +1 for temporal position
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output importance score in [0, 1]
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, frame_features: torch.Tensor, temporal_positions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            frame_features: Tensor of shape (batch_size, vision_dim)
            temporal_positions: Tensor of shape (batch_size, 1) with normalized positions [0, 1]
            
        Returns:
            importance_scores: Tensor of shape (batch_size, 1) with predicted importance scores
        """
        # Concatenate visual features with temporal position
        x = torch.cat([frame_features, temporal_positions], dim=1)  # (batch_size, vision_dim + 1)
        
        # Pass through MLP
        importance_scores = self.mlp(x)  # (batch_size, 1)
        
        return importance_scores

class StudentTrainer:
    """
    Trainer for the student frame importance predictor
    """
    def __init__(self, model: FrameImportancePredictor, teacher_model, device: str = "cuda:0"):
        self.model = model.to(device)
        self.teacher = teacher_model
        self.device = torch.device(device)
        
        # Freeze teacher's vision backbone for feature extraction
        for param in self.teacher.vision_backbone.parameters():
            param.requires_grad = False
        
        logger.info("Student trainer initialized")
    
    def extract_training_features(self, video_data: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract training features from video data
        
        Args:
            video_data: List of video data with frames and importance scores
            
        Returns:
            features: (N, vision_dim + 1) - frame features + temporal position
            positions: (N, 1) - temporal positions  
            labels: (N,) - importance scores
        """
        all_features = []
        all_positions = []
        all_labels = []
        
        logger.info("Extracting training features...")
        
        # Track actual feature dimension for the first video
        actual_vision_dim = None
        
        for video_info in video_data:
            frames = video_info['frames']  # (T, H, W, 3)
            scores = video_info['importance_scores']  # List of T scores
            
            if len(frames) == 0 or len(scores) == 0:
                continue
            
            # Debug: Check frame data format
            logger.info(f"Processing video: {video_info.get('title', 'Unknown')[:50]}...")
            logger.info(f"Frames shape: {frames.shape}, dtype: {frames.dtype}")
            logger.info(f"Frame value range: {frames.min()} - {frames.max()}")
            
            # Convert frames to PIL Images
            # Ensure frames are in correct format (uint8, 0-255 range)
            if frames.dtype != np.uint8:
                if frames.max() <= 1.0:
                    # Normalize from [0,1] to [0,255]
                    frames = (frames * 255).astype(np.uint8)
                else:
                    # Clip and convert to uint8
                    frames = np.clip(frames, 0, 255).astype(np.uint8)
            
            pil_frames = []
            for i, frame in enumerate(frames):
                try:
                    pil_frame = Image.fromarray(frame)
                    pil_frames.append(pil_frame)
                except Exception as e:
                    logger.error(f"Error converting frame {i} to PIL: {e}")
                    logger.error(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
                    continue
            
            # Extract visual features using teacher's vision backbone
            with torch.no_grad():
                frame_features = self.teacher.extract_frame_features(pil_frames)  # (T, vision_dim)
            
            # Check and update actual vision dimension
            if actual_vision_dim is None:
                actual_vision_dim = frame_features.shape[1]
                logger.info(f"Detected actual vision feature dimension: {actual_vision_dim}")
                
                # Update student model if dimension doesn't match
                if actual_vision_dim != self.model.vision_dim:
                    logger.info(f"Updating student model vision dimension from {self.model.vision_dim} to {actual_vision_dim}")
                    # Recreate the student model with correct dimensions
                    self.model = FrameImportancePredictor(actual_vision_dim, self.model.hidden_dim)
                    self.model.to(self.device)
            
            # Create temporal positions
            num_frames = len(frames)
            positions = torch.tensor([i / (num_frames - 1) if num_frames > 1 else 0.0 
                                    for i in range(num_frames)], dtype=torch.float32).unsqueeze(1)  # (T, 1)
            
            # Convert scores to tensor
            score_tensor = torch.tensor(scores, dtype=torch.float32).unsqueeze(1)  # (T, 1)
            
            all_features.append(frame_features)
            all_positions.append(positions)
            all_labels.append(score_tensor)
        
        # Concatenate all data
        features = torch.cat(all_features, dim=0)  # (N, vision_dim)
        positions = torch.cat(all_positions, dim=0)  # (N, 1)
        labels = torch.cat(all_labels, dim=0)  # (N, 1)
        
        logger.info(f"Extracted features for {len(features)} frames")
        return features, positions, labels
    
    def train(self, video_data: List[Dict[str, Any]], num_epochs: int = 10, 
              batch_size: int = 32, learning_rate: float = 3e-4, 
              val_split: float = 0.2) -> Dict[str, List[float]]:
        """
        Train the student model
        
        Args:
            video_data: List of video data with frames and importance scores
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            val_split: Fraction of data to use for validation
            
        Returns:
            Dictionary with training history
        """
        # Extract features and labels
        features, positions, labels = self.extract_training_features(video_data)
        
        # Split into train/validation
        num_samples = len(features)
        num_val = int(num_samples * val_split)
        indices = torch.randperm(num_samples)
        
        train_indices = indices[num_val:]
        val_indices = indices[:num_val]
        
        train_features = features[train_indices]
        train_positions = positions[train_indices]
        train_labels = labels[train_indices]
        
        val_features = features[val_indices]
        val_positions = positions[val_indices]
        val_labels = labels[val_indices]
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(train_features, train_positions, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = torch.utils.data.TensorDataset(val_features, val_positions, val_labels)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup optimizer and loss
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        mse_loss = nn.MSELoss()
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_mse': [],
            'val_mse': []
        }
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_losses = []
            train_mse_losses = []
            
            for batch_features, batch_positions, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_positions = batch_positions.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model(batch_features, batch_positions)
                
                # Compute losses
                mse = mse_loss(predictions, batch_labels)
                ranking_loss = self._compute_ranking_loss(predictions, batch_labels)
                
                # Combined loss
                total_loss = mse + 0.1 * ranking_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                train_losses.append(total_loss.item())
                train_mse_losses.append(mse.item())
            
            # Validation phase
            self.model.eval()
            val_losses = []
            val_mse_losses = []
            
            with torch.no_grad():
                for batch_features, batch_positions, batch_labels in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_positions = batch_positions.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    predictions = self.model(batch_features, batch_positions)
                    
                    mse = mse_loss(predictions, batch_labels)
                    ranking_loss = self._compute_ranking_loss(predictions, batch_labels)
                    total_loss = mse + 0.1 * ranking_loss
                    
                    val_losses.append(total_loss.item())
                    val_mse_losses.append(mse.item())
            
            # Record history
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            avg_train_mse = np.mean(train_mse_losses)
            avg_val_mse = np.mean(val_mse_losses)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_mse'].append(avg_train_mse)
            history['val_mse'].append(avg_val_mse)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                       f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                       f"Train MSE: {avg_train_mse:.4f}, Val MSE: {avg_val_mse:.4f}")
        
        logger.info("Training completed!")
        return history
    
    def _compute_ranking_loss(self, predictions: torch.Tensor, labels: torch.Tensor, 
                            margin: float = 0.05) -> torch.Tensor:
        """
        Compute pairwise ranking loss
        
        Args:
            predictions: Predicted importance scores (batch_size, 1)
            labels: True importance scores (batch_size, 1)
            margin: Margin for ranking loss
            
        Returns:
            Ranking loss
        """
        batch_size = predictions.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=predictions.device)
        
        # Create all pairs
        pred_diff = predictions.unsqueeze(1) - predictions.unsqueeze(0)  # (batch_size, batch_size)
        label_diff = labels.unsqueeze(1) - labels.unsqueeze(0)  # (batch_size, batch_size)
        
        # Only consider pairs where label difference > margin
        valid_pairs = (label_diff > margin).float()
        
        # Hinge loss: max(0, margin - pred_diff) for valid pairs
        hinge_loss = F.relu(margin - pred_diff) * valid_pairs
        
        # Average over valid pairs
        num_valid_pairs = valid_pairs.sum()
        if num_valid_pairs > 0:
            return hinge_loss.sum() / num_valid_pairs
        else:
            return torch.tensor(0.0, device=predictions.device)
    
    def predict_frame_importance(self, frames: List[Image.Image]) -> np.ndarray:
        """
        Predict importance scores for frames
        
        Args:
            frames: List of PIL Images
            
        Returns:
            importance_scores: Array of importance scores
        """
        self.model.eval()
        
        with torch.no_grad():
            # Extract features
            frame_features = self.teacher.extract_frame_features(frames)  # (T, vision_dim)
            
            # Create temporal positions
            num_frames = len(frames)
            positions = torch.tensor([i / (num_frames - 1) if num_frames > 1 else 0.0 
                                    for i in range(num_frames)], dtype=torch.float32).unsqueeze(1)  # (T, 1)
            
            # Move to device
            frame_features = frame_features.to(self.device)
            positions = positions.to(self.device)
            
            # Predict importance scores
            importance_scores = self.model(frame_features, positions)  # (T, 1)
            
            return importance_scores.cpu().numpy().flatten()
    
    def save_model(self, path: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vision_dim': self.model.vision_dim,
            'hidden_dim': self.model.hidden_dim
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")
