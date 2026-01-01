"""
Advanced Model Architectures for AHU Fault Detection
LSTM + CNN + Multi-Head Attention + Residual Connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Multi-head self-attention layer"""
    
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, embed_dim)
        Returns:
            x: (batch, seq_len, embed_dim)
        """
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        return x


class ResidualBlock(nn.Module):
    """Residual block with layer normalization"""
    
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.activation(self.norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.norm2(self.fc2(x))
        x = self.dropout(x)
        return self.activation(x + residual)


class HybridLSTMCNNAttention(nn.Module):
    """
    Hybrid model: LSTM + CNN + Multi-Head Attention
    
    Combines:
    - CNN: Extract local patterns from features
    - LSTM: Capture temporal dependencies
    - Attention: Focus on important features
    - Residual connections: Enable deeper networks
    
    Target: High-performance fault detection (~96%+ F1-score)
    Parameters: ~5-10M (CPU-friendly)
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        embed_dim: Embedding dimension (default: 256)
        cnn_channels: CNN channel progression (default: [128, 256, 384])
        lstm_hidden_dim: LSTM hidden dimension (default: 256)
        lstm_num_layers: Number of LSTM layers (default: 2)
        num_attention_heads: Number of attention heads (default: 8)
        num_residual_blocks: Number of residual blocks (default: 2)
        dropout: Dropout rate (default: 0.3)
    """

    def __init__(
        self,
        input_dim,
        num_classes,
        embed_dim=256,
        cnn_channels=[128, 256, 384],
        lstm_hidden_dim=256,
        lstm_num_layers=2,
        num_attention_heads=8,
        num_residual_blocks=2,
        dropout=0.3
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Input embedding
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # CNN Branch - Extract local patterns
        self.cnn_layers = nn.ModuleList()
        in_channels = 1
        for out_channels in cnn_channels:
            self.cnn_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout)
            ))
            in_channels = out_channels

        # Global pooling for CNN
        self.cnn_pool = nn.AdaptiveAvgPool1d(1)

        # LSTM Branch - Capture temporal dependencies
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=True
        )

        lstm_output_dim = lstm_hidden_dim * 2  # bidirectional

        # Multi-head attention on LSTM output
        self.attention = AttentionLayer(
            embed_dim=lstm_output_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )

        # Feature projection
        cnn_out_dim = cnn_channels[-1]
        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_out_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.lstm_proj = nn.Sequential(
            nn.Linear(lstm_output_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Fusion with residual blocks
        fusion_dim = embed_dim * 2  # CNN + LSTM
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Residual blocks for deep processing
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(512, dropout=dropout) for _ in range(num_residual_blocks)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            logits: Output logits of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)

        # Input projection
        x_embed = self.input_proj(x)  # (batch, embed_dim)

        # CNN Branch
        x_cnn = x.unsqueeze(1)  # (batch, 1, input_dim)
        for cnn_layer in self.cnn_layers:
            x_cnn = cnn_layer(x_cnn)
        x_cnn = self.cnn_pool(x_cnn).squeeze(-1)  # (batch, cnn_channels[-1])
        x_cnn = self.cnn_proj(x_cnn)  # (batch, embed_dim)

        # LSTM Branch
        x_lstm = x_embed.unsqueeze(1)  # (batch, 1, embed_dim)
        lstm_out, _ = self.lstm(x_lstm)  # (batch, 1, lstm_output_dim)

        # Apply attention
        x_lstm = self.attention(lstm_out)  # (batch, 1, lstm_output_dim)
        x_lstm = x_lstm.squeeze(1)  # (batch, lstm_output_dim)
        x_lstm = self.lstm_proj(x_lstm)  # (batch, embed_dim)

        # Fusion
        x_fused = torch.cat([x_cnn, x_lstm], dim=1)  # (batch, embed_dim*2)
        x_fused = self.fusion(x_fused)  # (batch, 512)

        # Apply residual blocks
        for res_block in self.residual_blocks:
            x_fused = res_block(x_fused)

        # Classification
        logits = self.classifier(x_fused)

        return logits


def create_model(input_dim, num_classes=4, model_type='hybrid', dropout=0.3, **kwargs):
    """
    Factory function to create models
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        model_type: 'hybrid' (currently only supported)
        dropout: Dropout rate
        **kwargs: Additional model parameters
        
    Returns:
        PyTorch model
    """
    if model_type == 'hybrid':
        return HybridLSTMCNNAttention(
            input_dim=input_dim,
            num_classes=num_classes,
            embed_dim=kwargs.get('embed_dim', 256),
            cnn_channels=kwargs.get('cnn_channels', [128, 256, 384]),
            lstm_hidden_dim=kwargs.get('lstm_hidden_dim', 256),
            lstm_num_layers=kwargs.get('lstm_num_layers', 2),
            num_attention_heads=kwargs.get('num_attention_heads', 8),
            num_residual_blocks=kwargs.get('num_residual_blocks', 2),
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported: 'hybrid'")


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")

    input_dim = 15  # Typical number of features
    num_classes = 4

    model = HybridLSTMCNNAttention(input_dim=input_dim, num_classes=num_classes)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} (~{total_params/1e6:.1f}M)")

    # Test forward pass
    x = torch.randn(32, input_dim)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("✓ Model works!")

