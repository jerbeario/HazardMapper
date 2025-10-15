"""HazardMapper - Architecture Module
========================
This module contains various neural network architectures for hazard susceptibility modeling.

Architectures included:
- SimpleCNN
- FullCNN
- MLP
- UNet (not functional yet)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init




# MLP to test 1d data 
class MLP(nn.Module):
    def __init__(self, logger, device, num_vars, n_layers=2, n_nodes=128, 
                 dropout=True, drop_value=0.4, patch_size=1):
        """
        MLP architecture designed to take feature vectors as input.
        
        Args:
            logger: Logger instance
            device: Torch device
            num_vars: Number of input variables/features
            n_layers: Number of hidden layers
            n_nodes: Number of nodes in each hidden layer
            dropout: Whether to use dropout
            drop_value: Dropout probability
            patch_size: Should be 1 for pure feature vector input
        """
        super(MLP, self).__init__()
        
        self.logger = logger
        self.device = device
        self.num_vars = num_vars
        
        # For feature vector input, input size is just num_vars
        # (when patch_size=1, we get features only with no spatial context)
        input_size = num_vars
        
        # Build the MLP layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, n_nodes))
        layers.append(nn.BatchNorm1d(n_nodes))
        layers.append(nn.ReLU())
        if dropout:
            layers.append(nn.Dropout(drop_value))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_nodes, n_nodes))
            layers.append(nn.BatchNorm1d(n_nodes))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(drop_value))
        
        # Output layer
        layers.append(nn.Linear(n_nodes, 1))
        # layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
        self.logger.info(f"Created MLP with {n_layers} layers, {n_nodes} nodes per layer")
        self.logger.info(f"Input size: {input_size}, Output size: 1")

    
    def forward(self, x):
        """
        Forward pass through the MLP.
        
        Args:
            x: Input tensor of shape [batch_size, num_vars, 1, 1]
            
        Returns:
            Output tensor of shape [batch_size, 1]
        """
        # Reshape input based on whether it's a single feature vector or patches
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_vars)
    
        # Forward pass through the model
        return self.model(x)

# Testing
class CNN_GAP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_layers: int = 2,
        n_filters: int = 32,
        drop_value: float = 0.3,
        n_nodes: int = 256,
        use_dropout: bool = True,
    ):
        super().__init__()

        # Convolutional block
        conv_layers = []
        current_channels = in_channels
        for i in range(n_layers):
            out_channels = n_filters * (2 ** i)
            conv_layers += [
                nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ]
            current_channels = out_channels

        self.conv_block = nn.Sequential(*conv_layers)

        # GAP-based classifier (patch-size agnostic)
        classifier_layers = [
            nn.AdaptiveAvgPool2d(1),   # → [B, C, 1, 1]
            nn.Flatten(),             # → [B, C]
            nn.Linear(current_channels, n_nodes),
            nn.ReLU()
        ]
        if use_dropout:
            classifier_layers.append(nn.Dropout(drop_value))
        classifier_layers.append(nn.Linear(n_nodes, 1))
        # classifier_layers.append(nn.Sigmoid())
        self.classifier = nn.Sequential(*classifier_layers)


    def forward(self, x):
        x = self.conv_block(x)
        x = self.classifier(x)
        return x

class CNN_GAPatt(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_layers: int = 2,
        n_filters: int = 32,
        drop_value: float = 0.3,
        n_nodes: int = 256,
        use_dropout: bool = True,  
    ):
        super().__init__()

        # Convolutional feature extractor
        conv_layers = []
        current_channels = in_channels
        for i in range(n_layers):
            out_channels = n_filters * (2 ** i)
            conv_layers += [
                nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ]
            current_channels = out_channels

        self.conv_block = nn.Sequential(*conv_layers)
        self.attn = SpatialAttentionLayer(current_channels) 

        # GAP-based classifier
        classifier_layers = [
            nn.AdaptiveAvgPool2d(1),  # → [B, C, 1, 1]
            nn.Flatten(),             # → [B, C]
            nn.Linear(current_channels, n_nodes),
            nn.ReLU(),
        ]
        if use_dropout:
            classifier_layers.append(nn.Dropout(drop_value))
        classifier_layers.append(nn.Linear(n_nodes, 1))
        # classifier_layers.append(nn.Sigmoid())
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.attn(x)
        x = self.classifier(x)
        return x
    
class CNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_layers: int = 2,
        n_filters: int = 32,
        drop_value: float = 0.3,
        patch_size: int = 5
    ):
        super().__init__()

        # Convolutional stack
        conv_layers = []
        current_channels = in_channels
        for i in range(n_layers):
            out_channels = n_filters * (2 ** i)
            conv_layers += [
                nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ]
            current_channels = out_channels

        self.conv_block = nn.Sequential(*conv_layers)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(current_channels * patch_size * patch_size, 256),
            nn.ReLU(),
            nn.Dropout(drop_value),
            nn.Linear(256, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_block(x)
        return self.classifier(x)

        
class CNNatt(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_layers: int = 2,
        n_filters: int = 32,
        drop_value: float = 0.3,
        patch_size: int = 5
    ):
        super().__init__()

        # Convolutional stack
        conv_layers = []
        current_channels = in_channels
        for i in range(n_layers):
            out_channels = n_filters * (2 ** i)
            conv_layers += [
                nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ]
            current_channels = out_channels

        self.conv_block = nn.Sequential(*conv_layers)

        # Spatial attention after conv stack
        self.attn = SpatialAttentionLayer(current_channels)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(current_channels * patch_size * patch_size, 256),
            nn.ReLU(),
            nn.Dropout(drop_value),
            nn.Linear(256, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.attn(x)
        return self.classifier(x)

# Simple CNN architecture to test the pipeline
class SimpleCNN(nn.Module):
    def __init__(self, logger, device, num_vars, filters=16, n_layers=2, dropout=False, 
                 drop_value=0.2, patch_size=5):
        """
        Simple CNN with constant filters and variable number of convolutional layers.
        
        Args:
            logger: Logger instance
            device: Torch device
            num_vars: Number of input variables/channels
            filters: Number of filters for all convolution layers
            num_layers: Number of shared convolutional layers after concatenation
            dropout: Whether to use dropout
            drop_value: Dropout probability
            patch_size: Size of the input neighborhood
        """
        super(SimpleCNN, self).__init__()
        
        self.logger = logger
        self.device = device
        self.num_vars = num_vars
        self.n_layers = n_layers
        
        # Per-variable feature extractors
        self.feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, filters, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(filters),
            )
            for _ in range(num_vars)
        ])
        
        # Build shared conv block with constant filters
        shared_convs = []
        in_channels = filters * num_vars
        for i in range(n_layers):
            out_channels = filters  # Constant filters for all layers
            shared_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            shared_convs.append(nn.ReLU())
            shared_convs.append(nn.BatchNorm2d(out_channels))

            in_channels = out_channels
        
        self.shared_conv = nn.Sequential(*shared_convs)
        # self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d((1))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters, 1024),  # Updated to match constant filters
            nn.ReLU(),
            nn.Dropout(drop_value) if dropout else nn.Identity(),
            nn.Linear(1024, 1),
            # nn.Sigmoid()
        )
    
    def forward(self, x):
        # Split input by variables
        var_inputs = [x[:, i:i+1] for i in range(self.num_vars)]
        
        # Extract features
        features = [extractor(var_input) for extractor, var_input in zip(self.feature_extractors, var_inputs)]
        
        # Concatenate and pass through shared convs
        x = torch.cat(features, dim=1)
        x = self.shared_conv(x)
        # x = self.pool(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        
        return x


# model from Japan paper converted to pytorch
class SpatialAttentionLayer(nn.Module):
    def __init__(self, channels, device=None):
        super(SpatialAttentionLayer, self).__init__()
        self.device = device
        
        self.conv1x1_theta = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.conv1x1_phi = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.conv1x1_g = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        # Initialize weights
        init.kaiming_normal_(self.conv1x1_theta.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.conv1x1_phi.weight, mode='fan_in', nonlinearity='relu')
        init.xavier_uniform_(self.conv1x1_g.weight)

        if self.device is not None:
            self.conv1x1_theta = self.conv1x1_theta.to(self.device)
            self.conv1x1_phi = self.conv1x1_phi.to(self.device)
            self.conv1x1_g = self.conv1x1_g.to(self.device)

    def forward(self, x):
        theta = F.relu(self.conv1x1_theta(x))
        phi = F.relu(self.conv1x1_phi(x))
        g = torch.sigmoid(self.conv1x1_g(x))

        theta_phi = theta * phi
        attention = theta_phi * g
        attended_x = x + attention
        
        return attended_x
    
class SpatialAttentionCNN(nn.Module):
    def __init__(self, logger, device, num_vars, filters=16, n_layers=2, dropout=False, 
                 drop_value=0.2, patch_size=5):
        """
        Simple CNN with constant filters, variable number of conv layers, and spatial attention.
        """
        super(SpatialAttentionCNN, self).__init__()
        
        self.logger = logger
        self.device = device
        self.num_vars = num_vars
        self.n_layers = n_layers
        
        # Per-variable feature extractors
        self.feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, filters, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(filters),

            )
            for _ in range(num_vars)
        ])
        
        # Shared conv block with constant filters
        shared_convs = []
        in_channels = filters * num_vars
        for i in range(n_layers):
            out_channels = filters
            shared_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            shared_convs.append(nn.ReLU())
            shared_convs.append(nn.BatchNorm2d(out_channels))

            in_channels = out_channels
        
        self.shared_conv = nn.Sequential(*shared_convs)
        
        # Add Spatial Attention Layer
        self.spatial_attention = SpatialAttentionLayer(channels=filters, device=self.device)        
        # self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters, 1024),
            nn.ReLU(),
            nn.Dropout(drop_value) if dropout else nn.Identity(),
            nn.Linear(1024, 1),
            # nn.Sigmoid()
        )
    
    def forward(self, x):
        var_inputs = [x[:, i:i+1] for i in range(self.num_vars)]
        features = [extractor(var_input) for extractor, var_input in zip(self.feature_extractors, var_inputs)]
        
        x = torch.cat(features, dim=1)
        x = self.shared_conv(x)
        x = self.spatial_attention(x)  # Apply spatial attention
        # x = self.pool(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        
        return x