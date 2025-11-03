""" HazardMapper - Model module.
==========================
This module contains the Baseline and HazardModel classes for training and evaluating hazard susceptibility models.
A ModelMgr class is also included to handle the training and evaluation of different model architectures.
"""

from matplotlib import ticker
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import seaborn as sns
import numpy as np
import pandas as pd
import os

import logging
import logging.handlers

from joblib import dump, load
import sys
import time
import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, precision_score, recall_score, f1_score, 
    roc_auc_score, precision_recall_curve, 
    average_precision_score, roc_curve, log_loss
    )

import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from torch.nn import BCELoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
import torch.multiprocessing 
from torchsummary import summary

import wandb

from HazardMapper.dataset import HazardDataset, raw_paths, var_paths, label_paths, label_paths_downscaled, var_paths_downscaled, partition_paths, partition_paths_downscaled
from HazardMapper.architecture import MLP, CNN, SimpleCNN, SpatialAttentionCNN, CNNatt, CNN_GAP, CNN_GAPatt
from HazardMapper.utils import plot_npy_arrays
from HazardMapper import ROOT

import numpy as np
import shap


# plt.style.use('bauhaus_light')






class Baseline:
    """
    Baseline model class for training and evaluating traditional ML models (Random Forest, Logistic Regression).
    Only supports patch_size=1 (pixel-wise classification). 

    """
    def __init__(self, hazard, architecture, variables, output_path, logger, seed):

        self.hazard = hazard.lower()
        self.architecture = architecture
        self.output_path = output_path
        self.seed = seed
        self.logger = logger
        self.variables = variables
        self.num_vars = len(variables)


        # Check if the model architecture is valid
        if self.architecture not in ["RF", "LR", "MLP"]:
            raise ValueError(f"Baseline architecture '{self.architecture}' is not defined.")

    def numpy_dataset(self, mask):
        """
        Create dataset from 2D mask.

        Args:
            mask: 2D boolean array indicating valid pixels
        """
        # Check if the hazard is valid
        if self.hazard not in label_paths.keys():
            raise ValueError(f"Hazard {self.hazard} not found in label_paths")
        
        # Load labels as 2D array, keep spatial structure
        
        labels_2d = np.load(label_paths[self.hazard])
        labels_binary = (labels_2d > 0).astype(int)
        
        # Load features for each variable as 2D
        feature_arrays = []
        for variable in self.variables:
            if variable not in var_paths.keys():
                raise ValueError(f"Variable {variable} not found in var_paths")
            var_data = np.load(var_paths[variable])
            feature_arrays.append(var_data)
        
        # Stack features along last dimension: (H, W, num_vars)
        X = np.stack(feature_arrays, axis=-1)
        
        # Apply mask to get valid samples
        X = X[mask]  # Shape: (N, num_vars)
        y = labels_binary[mask]  # Shape: (N,)

        return X, y

    def load_data(self):
        """
        Load training and testing data based on partition maps.

        Returns:
            X_train_val, y_train_val: Training/validation features and labels
            X_test, y_test: Testing features and labels
        """
        self.logger.info(f"Loading data...")
        # Load partition map as 2D
        partition_path = partition_paths[self.hazard]
        partition_map = np.load(partition_path)

        train_val_mask = (partition_map == 1) | (partition_map == 2)
        test_mask = (partition_map == 3)

        # Make dataset from masks instead of indices
        X_train_val, y_train_val = self.numpy_dataset(train_val_mask)
        X_test, y_test = self.numpy_dataset(test_mask)

        # Log dataset info
        self.logger.info(f"Train/Val data shape: {X_train_val.shape}, Labels shape: {y_train_val.shape}")
        self.logger.info(f"Class balance in train/val set: {np.bincount(y_train_val)}")
        self.logger.info(f"Test data shape: {X_test.shape}, Labels shape: {y_test.shape}")
        self.logger.info(f"Class balance in test set: {np.bincount(y_test)}")
        
        return X_train_val, y_train_val, X_test, y_test
       
    def design_model(self):
        """
        Define the model based on the specified architecture.
        """
        self.logger.info(f"Designing {self.architecture} model...")
        
        if self.architecture == "RF":
            model = RandomForestClassifier(
                n_estimators=100, #100
                max_depth=10, #10
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                random_state=self.seed,
                n_jobs=-1,  # Use all available cores
            )
        elif self.architecture == "LR":
            model = LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="sag",
                max_iter=100,
                random_state=self.seed,
            )
        
        else:
            raise ValueError(f"Model '{self.model}' is not defined. Choose 'RF' or 'LR'.")
        
        return model

    def train(self):
        """
        Train the model using the training data.
        """        
        file_name = f'{self.architecture}_{self.hazard}.joblib'
        self.logger.info(f"Training {self.architecture} model...")
        self.model.fit(self.X_train, self.y_train)

        # Check if the output directory exists, if not create it
        self.logger.info('Saving model...')

        # Save the model
        dump(self.model, f"{self.output_path}{file_name}")
        self.logger.info(f"Model saved to {self.output_path}{file_name}")    

    def testing(self):
        """
        Test the model on the test set.

        Returns:
            y_test: True labels
            y_pred_proba: Predicted probabilities for the positive class
        """
        self.logger.info('Testing model...')
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]

        return self.y_test, y_pred_proba
    
    def predict(self, mask):
        """ 
        Predict probabilities using the trained model on the given mask

        Args:
            mask: 2D boolean array indicating valid pixels
        Returns:
            y_pred_proba: Predicted probabilities for the positive class
        """
        self.logger.info('Predicting using the trained model...')
        X, _ = self.numpy_dataset(mask)
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        return y_pred_proba

    def main(self):
        """
        Main function to run the training and testing pipeline.

        Returns:
            y_true: True labels from the test set
            y_prob: Predicted probabilities for the positive class from the test set
        """
        self.X_train, self.y_train, self.X_test, self.y_test = self.load_data()
        self.model = self.design_model()
        self.train()
        y_true, y_prob = self.testing()
        return y_true, y_prob
    
class HazardModel:
    """
    HazardModel class for training and evaluating deep learning models using PyTorch.
    Supports architectures: 'MLP', 'CNN', 'SimpleCNN', 'SpatialAttentionCNN' with variable patch sizes.

    """
    def __init__(self, hazard, variables, patch_size, batch_size, architecture, 
                 logger, seed, epoch, early_stopping, patience, min_delta, output_dir):
        super(HazardModel, self).__init__()

        # Configs
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        self.hazard = hazard

        # Model parameters
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_vars = len(variables)
        self.variables = variables
        self.seed = seed
        self.name_model = hazard + '_' + architecture
        self.architecture = architecture
        self.best_model_state = None

        # Data
        self.num_workers = 1 # number of workers for data loading
        
        # Hyperparameters
        self.learning_rate = 0.002
        self.weight_decay = 0.00005 # L2 regularization
        self.filters = 64 #64
        self.n_layers = 1 #1
        self.drop_value =  0.3 #0.4
        self.kernel_size = 3
        self.pool_size = 2  
        self.dropout = True
        self.n_nodes = 128 # for MLP architecture

        # Training
        self.loss_fn = BCEWithLogitsLoss()
        self.current_epoch = 0
        self.epochs = epoch
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.average_epoch_time = 0


        # outputs
        self.output_dir = output_dir



    def design_basemodel(self):
        """
        Define architecture in PyTorch.
        Currently supports 'MLP', 'CNN', 'SimpleCNN', and 'SpatialAttentionCNN'.

        Returns:
            model: PyTorch model instance

        """
        if self.architecture == 'MLP':

            model = MLP(
                logger=self.logger,
                device=self.device,
                num_vars=self.num_vars,
                n_layers=self.n_layers,
                n_nodes=2*self.filters, 
                dropout=self.dropout,
                drop_value=self.drop_value,
                patch_size=self.patch_size
            )

        elif self.architecture == 'CNN':

            model = CNN(
                in_channels=self.num_vars,
                n_filters=self.filters,
                n_layers=self.n_layers,
                drop_value=self.drop_value,
                patch_size=self.patch_size,

            )

        elif self.architecture == 'CNNatt':
            model = CNNatt(
                in_channels=self.num_vars,
                n_filters=self.filters,
                n_layers=self.n_layers,
                drop_value=self.drop_value,
                patch_size=self.patch_size,
            )
        elif self.architecture == 'CNN_GAP':
            model = CNN_GAP(
                in_channels=self.num_vars,
                n_filters=self.filters,
                n_layers=self.n_layers,
                drop_value=self.drop_value,
                n_nodes=self.n_nodes,
                use_dropout=self.dropout,

            )
            
        elif self.architecture == 'CNN_GAPatt':
            model = CNN_GAPatt(
                in_channels=self.num_vars,
                n_filters=self.filters,
                n_layers=self.n_layers,
                drop_value=self.drop_value,
                n_nodes=self.n_nodes,
                use_dropout=self.dropout,
            )

        elif self.architecture == 'SimpleCNN':

            model = SimpleCNN(
                logger=self.logger,
                device=self.device,
                num_vars=self.num_vars,
                filters=self.filters,
                dropout=self.dropout,
                drop_value=self.drop_value,
                patch_size=self.patch_size,
                n_layers=self.n_layers,
            )

        elif self.architecture == 'SpatialAttentionCNN':
            model = SpatialAttentionCNN(
                device=self.device,
                logger=self.logger,
                num_vars=self.num_vars,
                filters=self.filters,
                # kernel_size=self.kernel_size,
                # pool_size=self.pool_size,
                n_layers=self.n_layers,
                dropout=self.dropout,
                drop_value=self.drop_value,
                # name_model=self.name_model,
                patch_size=self.patch_size
            )


        self.model = model
        self.model.to(self.device)

        self.logger.info(f'''Model architecture: {self.architecture}\n
                            Number of variables: {self.num_vars}\n
                            Patch size: {self.patch_size}\n
                            Batch size: {self.batch_size}\n
                            Learning rate: {self.learning_rate}\n
                            Filters: {self.filters}\n
                            Number of layers: {self.n_layers}\n
                            Dropout value: {self.drop_value}\n
                            Weight decay: {self.weight_decay}\n
                            Dropout: {self.dropout}\n
                            Kernel size: {self.kernel_size}\n
                            Pool size: {self.pool_size}\n
                            Nodes in MLP: {self.n_nodes}\n

                          ''')
        if self.device.type != 'mps':  # torchsummary does not support MPS
            summary(self.model, input_size=(self.num_vars, self.patch_size, self.patch_size), device=str(self.device))
        else:
            # print number of parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            self.logger.info(f"Total parameters: {total_params:,}")
            self.logger.info(f"Trainable parameters: {trainable_params:,}\n")

        return model


    def load_dataset(self):
        """
        Preprocess the data for the model, using the dataset class.
        
        Returns:
            train_loader, val_loader, test_loader: DataLoader instances for training, validation, and testing
        """
        
        # loading partition map 
        # TODO generalize for other hazard partition maps
        self.logger.info('Loading partition map')
        partition_map = np.load(partition_paths[self.hazard])


        partition_shape = partition_map.shape
        self.logger.info(f'Splitting dataset into train, validation and test sets')

        dataset = HazardDataset(hazard=self.hazard, variables=self.variables, patch_size=self.patch_size, downscale=downscale)
        
        idx_transform = np.array([[partition_shape[1]],[1]])

        train_indices = (np.argwhere(partition_map == 1) @ idx_transform).flatten()
        val_indices = (np.argwhere(partition_map == 2) @ idx_transform).flatten()
        test_indices = (np.argwhere(partition_map == 3) @ idx_transform).flatten()

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)
        
        train_loader = DataLoader(train_dataset, num_workers=self.num_workers, batch_size= self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=False)

        self.logger.info(f"Train dataset size: {len(train_indices)}")
        self.logger.info(f"Validation dataset size: {len(val_indices)}")
        self.logger.info(f"Test dataset size: {len(test_indices)}")

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
    
        return train_loader, val_loader, test_loader

    def train(self):
        """
        Train the model using the provided data loaders.

        1. Sets up optimizer, loss function, and learning rate scheduler.
        2. Loops over epochs, performing training and validation.
        3. Implements early stopping based on validation loss.
        4. Saves the best model based on validation loss.
        5. Logs metrics to Weights & Biases (wandb).

        """
        self.logger.info(f'Training the model...') 
        # Always attempt to watch model in W&B (manager ensures a run is active)
        try:
            wandb.watch(self.model, log="all")
        except Exception:
            # don't fail training if wandb is not available for some reason
            self.logger.debug("wandb.watch() failed or no active run")

        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, ) # added L2 regularization
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.8,
            patience=self.patience,
        )
        # Define loss function
       


        # Initialize trackers
        self.epochs_without_improvement = 0
        self.best_val_loss = float('inf')
        start_time = time.time()


        # Training loop
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            self.model.train()
            train_loss = 0.0
            train_mae = 0.0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Reshape Labels to match output 
                labels = labels.unsqueeze(1)

                # Forward pass with sigmoid activation
                outputs = self.model(inputs)
                # outputs = torch.sigmoid(outputs)


                # loss = F.binary_cross_entropy(outputs, labels, weight=sample_weigts )
                loss = self.loss_fn(outputs, labels)

                mae = self.safe_mae(labels, torch.sigmoid(outputs))

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Step the optimizer
                optimizer.step()

                train_loss += loss.item()
                train_mae += mae.item()

            
            # Calculate average metrics
            train_loss /= len(self.train_loader)
            train_mae /= len(self.train_loader)

            
            # Evaluate on validation data
            val_loss, val_mae = self.evaluate()

            # Log epoch metrics
            self.logger.info(f"———Epoch {epoch+1}/{self.epochs}———")
            self.logger.info(f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}")
            self.logger.info(f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")

            # Save the best model
            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                self.save_best_model()
                self.epochs_without_improvement = 0
                self.logger.info(f"New best model saved (lowest validation loss: {self.best_val_loss:.4f})")

            else:
                self.epochs_without_improvement += 1
                self.logger.info(f"  No improvement for {self.epochs_without_improvement} epochs")
                
                # Early stopping check
                if self.early_stopping and self.epochs_without_improvement >= self.patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break

            # Update learning rate
            scheduler.step(val_loss)
            self.learning_rate = scheduler.get_last_lr()[0]

            end_time = time.time()
            training_time = end_time - start_time 
            self.average_epoch_time = training_time / (epoch + 1)

            # Log metrics to wandb
            try:
                wandb.log({
                     "epoch": epoch,
                     "train_loss": train_loss,
                     "train_MAE": train_mae,
                     "val_loss": val_loss,
                     "val_MAE": val_mae,
                     "learning_rate": self.learning_rate,
                     "epoch_time": self.average_epoch_time,
                 })
            except Exception:
                self.logger.debug("wandb.log() failed for epoch metrics")
        try:
            wandb.unwatch(self.model)
        except Exception:
            self.logger.debug("wandb.unwatch() failed or no active run")

    @staticmethod
    def safe_mae(y_true, y_pred):
        """
        Mean absolute error (MAE) loss with NaN handling.
        """
        y_pred = torch.nan_to_num(y_pred, nan=0.0)
        y_true = torch.nan_to_num(y_true, nan=0.0)
        return torch.mean(torch.abs(y_pred - y_true))

    def save_best_model(self):
        # Create directory if it doesn't exist

        model_path = f"{self.output_dir}models/{self.best_val_loss:4f}_{wandb.run.id}.pth"
        onnx_path = f"{self.output_dir}models/{self.best_val_loss:4f}_{wandb.run.id}.onnx"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

        # Save PyTorch model
        torch.save(self.model.state_dict(), model_path)
        self.best_model_state = self.model.state_dict().copy()
        
        # Export to ONNX format
        dummy_input = torch.randn(1, self.num_vars, self.patch_size, self.patch_size).to(self.device)
        try:
            torch.onnx.export(
                self.model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            self.logger.info(f"ONNX model saved to {onnx_path}")
            # attempt to save artifact to W&B (manager ensures run exists)
            try:
                wandb.save(onnx_path)
            except Exception:
                self.logger.debug("wandb.save() failed for ONNX")
        except Exception as e:
            self.logger.error(f"Failed to export ONNX model: {e}")

    def load_best_model(self): 
        """
        Load the best model state from the saved file.
        """
        if self.best_model_state is not None:
            # self.design_basemodel()  # Recreate the model architecture
            self.model.load_state_dict(self.best_model_state)
            self.logger.info("Best model state loaded successfully.")
        else:
            # If no best model state is available, check if a saved model exists
            model_dir = f"{self.output_dir}/models"
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]

            if model_files:
                # find lowest validation loss model
                best_model_file = min(model_files, key=lambda f: float(f.split('_')[0]))
                model_path = f'{model_dir}/{best_model_file}'
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.logger.info(f"Best model loaded from {model_path}")
            else:
                # If no saved model exists, log an error
                self.logger.error("No saved model found. Please train the model first or provide a valid path.")
         
    def evaluate(self):
        """
        Evaluate the model on the provided data loader.

        Returns:
            val_loss: Average validation loss
            val_mae: Average validation mean absolute error (MAE)

        """
        self.model.eval()
        val_loss = 0.0
        val_mae = 0.0



        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.unsqueeze(1)

                # Forward pass
                outputs = self.model(inputs)
                # outputs = torch.sigmoid(outputs)
                
                # Calculate metrics
                # loss = self.safe_binary_crossentropy(labels, outputs)
                loss = self.loss_fn(outputs, labels)
                mae = self.safe_mae(labels, torch.sigmoid(outputs))

                
                val_loss += loss.item()
                val_mae += mae.item()


        # Calculate average metrics
        val_loss /= len(self.val_loader)
        val_mae /= len(self.val_loader)
        

        return val_loss, val_mae

    def test(self):
        """
        Test the model on the test set.

        Returns:
            y_true: True labels
            y_prob: Predicted probabilities for the positive class
        """
    
        self.logger.info('Loading best model for testing...')
        self.load_best_model()
        self.model.eval()

        y_true, y_prob = [], []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # outputs = self.model(inputs).squeeze()
                outputs = torch.sigmoid(self.model(inputs)).squeeze()
                y_true.extend(labels.cpu().numpy())
                y_prob.extend(outputs.cpu().numpy())

        y_true, y_prob = np.array(y_true), np.array(y_prob)

        return y_true, y_prob
    
    def predict(self, mask):
        """ 
        Creates Dataloader from mask and predicts using the model

        Args:
            mask: 2D boolean array indicating valid pixels
        Returns:
            predictions: 1D array of predicted probabilities for the positive class
        """
        # Get 1D indices from 2d mask
        mask_flat = mask.flatten()
        indices = np.argwhere(mask_flat).flatten()

        # Create a Subset of the dataset using the indices
        dataset = HazardDataset(hazard=self.hazard, variables=self.variables, patch_size=self.patch_size, downscale=downscale)        
        dataset = Subset(dataset, indices)  
        loader = DataLoader(dataset, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=False)
        self.logger.info(f"Dataset size: {len(dataset)}")   

        #predict using the model
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(self.device)
                outputs = torch.sigmoid(self.model(inputs)).squeeze()
                predictions.extend(outputs.cpu().numpy())

        return np.array(predictions)

    def load_model(self, model_path=None):
        """
        Load a pre-trained model from the specified path.
        
        Args:
            model_path (str): Path to the model file.
        """
        if model_path is None:
            # find model paths
            model_paths = [f for f in os.listdir(f"{self.output_dir}/models") if f.endswith('.pth')]

            # Sort paths by first part of the filename (validation loss)

            model_paths.sort(key=lambda f: float(f.split('_')[0]))

            model_path = f"{self.output_dir}/models/{model_paths[0]}"

        self.logger.info(f"Loading model from {model_path}")
        self.model = self.design_basemodel()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.logger.info("Model loaded successfully.")

    def explain_model(self, sample_size=1000, background_size=1000):
        """
        Compute SHAP values for the current model on a subset of samples.
        
        This example uses SHAP's DeepExplainer for PyTorch models.
        
        Args:
            sample_size (int): Number of test samples to explain.
            background_size (int): Number of background samples for SHAP.
            
        Returns:
            shap_values: The SHAP values computed for the sample.
        """


        # self.load_model()
        self.model.eval()

        # Retrieve background and sample batches
        background_batch, _ = next(iter(self.train_loader))
        background = background_batch[:background_size].to(self.device)
 
        sample_batch, _ = next(iter(self.test_loader))
        x_sample = sample_batch[:sample_size].to(self.device)

        # Compute SHAP values
        explainer = shap.DeepExplainer(self.model, background)
        shap_values = explainer.shap_values(x_sample, check_additivity=False)  # shape: [samples, channels, H, W]

        # ----- Channel-level summary plot -----
        channel_shap = np.mean(shap_values, axis=(2, 3, 4))  # [samples, channels]
        channel_inputs = np.mean(x_sample.cpu().numpy(), axis=(2, 3))  # [samples, channels]

        shap.summary_plot(
            channel_shap,
            features=channel_inputs,
            feature_names=self.variables,
            show=False,
            plot_size=(10, 6)
        )

        # 2) grab the current axes
        ax = plt.gca()

        # 3) set “nice” x-tick locations
        #    here we ask for at most 7 ticks, evenly spaced
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=7))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

        # 4) style the grid and labels
        ax.tick_params(axis='x', which='major', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_xlabel('SHAP value')

        # 5) title and layout
        plt.title(f"{self.hazard.capitalize()} SHAP Summary Plot", )
        plt.tight_layout()

        # 6) save and close
        plt.savefig(f"{self.output_dir}/{self.hazard}_shap_summary_plot.png", dpi=300)
        plt.close()

        # ----- Pixel-level average contribution plot -----
        # Average across samples and channels to get [H, W]
        shap_values_np = np.array(shap_values)  # [samples, channels, H, W]
        mean_pixel_contrib = np.mean(shap_values_np, axis=(0))  # [channels, H, W]
        max_pixel_contrib = np.max(abs(mean_pixel_contrib), axis=(0))  # [H, W]

        # Plot the maximum absolute pixel contribution
        plt.figure(figsize=(10, 10))
        plt.imshow(max_pixel_contrib, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Max Absolute Contribution')
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        plt.title(f"{self.hazard} Max Absoulte Pixel Contribution")
        plt.axis("off")
        plt.savefig(f"{self.output_dir}/{self.hazard}_shap_pixel_contribution.png")
        plt.close()

        # Log to wandb
        try:
            wandb.log({
                "shap_summary_plot": wandb.Image(f"{self.output_dir}/{self.hazard}_shap_summary_plot.png"),
                "shap_pixel_contribution": wandb.Image(f"{self.output_dir}/{self.hazard}_shap_pixel_contribution.png"),
            })
        except Exception:
            self.logger.debug("wandb.log() failed for SHAP plots")

    def main(self):
        """
        Main function to run the training and testing pipeline.
        Returns:
            y_true: True labels from the test set
            y_prob: Predicted probabilities for the positive class from the test set
        """

        self.load_dataset()
        self.design_basemodel()
        self.train()
        y_true, y_prob = self.test()
        return y_true, y_prob

class ModelMgr:
    """
    Model Manager class to handle different model architectures and training/evaluation. 
    Supports architectures: 'LR', 'RF', 'CNN', 'SimpleCNN', 'SpatialAttentionCNN', and 'MLP'. 
    Initializes model parameters, logger, and output directories.

    """
    def __init__(self, batch_size=1024, patch_size=5, architecture='CNN', 
            hazard='wildfire', epoch = 5, experiement_name='Experiment 1'):
        
        self.early_stopping = True
        self.patience = 4
        self.min_delta = 0.001
        self.hazard = hazard
        self.batch_size = batch_size
        self.missing_data_value = 0
        self.patch_size = patch_size
        self.architecture = architecture # 'CNN' or  or 'SimpleCNN' or 'SpatialAttentionCNN' or 'MLP'
        self.experiement_name = experiement_name
        self.epoch = epoch
        self.seed = 43
        self.logger, self.ch = self.set_logger()

        if self.hazard == 'landslide':
            self.variables = ['elevation', 'slope', 'landcover', 'aspect', 'NDVI', 'precipitation_daily', 'accuflux', 'HWSD', 'GEM', 'curvature', 'GLIM']
            
        elif self.hazard == 'flood':
            self.variables = ['elevation', 'slope', 'landcover', 'aspect', 'NDVI', 'precipitation_daily', 'accuflux']
    
        elif self.hazard == "wildfire":
            # temperature_daily, NDVI, landcover, elevation, wind_speed, fire_weather, soil_moisture(root or surface)
            self.variables = ['temperature_daily', 'NDVI', 'landcover', 'elevation', 'wind_speed_daily', 'fire_weather', 'soil_moisture_root']
        
        elif self.hazard == 'test':
            # temperature_daily, NDVI, landcover, elevation, wind_speed, fire_weather, soil_moisture(root or surface)
            self.variables = ['test']
        
        elif self.hazard == 'multi_hazard':
            self.variables = ['drought', 'extreme_wind', 'heatwave', 'volcano', 'earthquake', 'wildfire', 'landslide', 'flood']
        
        else:
            raise ValueError(f"Unknown hazard: {self.hazard}, choose from ['landslide', 'flood', 'wildfire']")
        
        if self.architecture in ['LR', 'RF']:
            self.model_type = 'baseline'
        elif self.architecture in ['CNN', 'SimpleCNN', 'SpatialAttentionCNN', 'MLP', 'CNNatt', 'CNN_GAP', 'CNN_GAPatt']:
            self.model_type = 'hazardmodel'
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}, choose from ['LR', 'RF', 'CNN', 'SimpleCNN', 'SpatialAttentionCNN', 'MLP', 'CNNatt', 'CNN_GAP']")


      
        # Create output directory if it doesn't exist
        self.output_dir = self.make_folder_structure()

 
    def build_model(self):
        """
        Build and return the appropriate model instance based on the architecture type.
        
        Returns:
            model_instance: Instance of the selected model class
        """
        if self.model_type == 'hazardmodel':
            hazard_model_instance = HazardModel(
                hazard=self.hazard,
                variables=self.variables,
                patch_size=self.patch_size,
                batch_size=self.batch_size,
                architecture=self.architecture,
                logger=self.logger,
                seed=self.seed,
                epoch=self.epoch,
                early_stopping=self.early_stopping,
                patience=self.patience,
                min_delta=self.min_delta,
                output_dir=self.output_dir,
            )
        elif self.model_type == 'baseline':
            hazard_model_instance = Baseline(
                hazard=self.hazard,
                variables=self.variables,
                architecture=self.architecture,
                logger=self.logger,
                seed=self.seed,
                output_path=self.output_dir,
            )


       
        return hazard_model_instance
        
    def set_logger(self, verbose=True):
        """
        Set-up the logging system, exit if this fails
        """
        # assign logger file name and output directory
        datelog = time.ctime()
        datelog = datelog.replace(':', '_')
        reference = f'{self.architecture}_{self.hazard}_Europe'

        logfilename = ('logger' + os.sep + reference + '_logfile_' + 
                    str(datelog.replace(' ', '_')) + '.log')

        # create output directory if not exists
        if not os.path.exists('logger'):
            os.makedirs('logger')

        # create logger and set threshold level, report error if fails
        try:
            logger = logging.getLogger(reference)
            logger.setLevel(logging.DEBUG)
        except IOError:
            sys.exit('IOERROR: Failed to initialize logger with: ' + logfilename)

        # set formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s -'
                                    '%(levelname)s - %(message)s')

        # assign logging handler to report to .log file
        ch = logging.handlers.RotatingFileHandler(logfilename,
                                                maxBytes=10*1024*1024,
                                                backupCount=5)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # assign logging handler to report to terminal
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        console.setFormatter(formatter)
        logger.addHandler(console)

        # start up log message
        logger.info('File logging to ' + logfilename)

        return logger, ch 
    
    def make_folder_structure(self):
        """
        Create the folder structure for saving outputs.
        Returns:
            output_dir (str): Path to the output directory.
        """
        
        # Create output directory if it doesn't exist
        output_dir = f'{ROOT}/Output/Europe/{self.hazard}/{self.architecture}/'
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Output directory created at: {output_dir}")
        
        return output_dir

    def evaluate(self, y_true, y_prob):
        """
        Evaluate model predictions with optimized threshold, compute metrics, and export results.

        Args:
            y_true (np.ndarray): True binary labels.
            y_prob (np.ndarray): Predicted probabilities.
            output_path (str): Path to save outputs.
            model_name (str): Name of the model (for output file naming).
            model_info (dict): Additional info (e.g., hyperparameters) to include in results.
            use_wandb (bool): Whether to log to Weights & Biases.
            wandb_run_name (str): Optional run name for W&B.

        Returns:
            metrics (dict): Computed performance metrics.


        """


        output_path = f'Output/Europe/{self.hazard}/evaluation'
        os.makedirs(output_path, exist_ok=True)
        model_name = f'{self.hazard}_{self.architecture}'

        # Save predictions to npy
        npy_path = os.path.join(output_path, f'{model_name}_predictions.npy')
        np.save(npy_path, y_prob)
        self.logger.info(f"Predictions saved to {npy_path}")
        # Save true labels to npy
        true_path = os.path.join(output_path, f'true_labels.npy')
        np.save(true_path, y_true)

       

        # Optimize threshold based on best F1
        precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5


        # Get predictions with best threshold
        y_pred = (y_prob >= best_threshold).astype(int)

        # Compute metrics
        metrics = {
            'Model': model_name,
            'Experiment': self.experiement_name,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1': f1_score(y_true, y_pred, zero_division=0),
            'AUROC': roc_auc_score(y_true, y_prob),
            'AP': average_precision_score(y_true, y_prob),
            'MAE': mean_absolute_error(y_true, y_prob),
            'BCE': log_loss(y_true, y_prob),
            'Best_Threshold': best_threshold
        }

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_path = os.path.join(output_path, f'{model_name}_roc_curve.png')
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUROC = {metrics["AUROC"]:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig(roc_path)
        plt.close()

        # Plot Precision-Recall curve
        pr_path = os.path.join(output_path, f'{model_name}_pr_curve.png')
        plt.figure()
        plt.plot(recall_curve, precision_curve, label=f'AP = {metrics["AP"]:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.savefig(pr_path)
        plt.close()

    
        # Plot output distribution
        bin_edges = np.arange(0, 1.005, 0.05)

        xy = np.concatenate((y_true.reshape(-1, 1), y_prob.reshape(-1, 1)), axis=1)

        pos = xy[xy[:, 0] == 1]
        neg = xy[xy[:, 0] == 0]

        pd_path = os.path.join(output_path, f'{model_name}_probability_distribution.png')
        fig, ax = plt.subplots(1, 2, figsize=(10, 6))
        bin_edges = np.arange(0, 1.1, 0.1)

        sns.histplot(neg[:, 1], bins=bin_edges, label="Predicted", color="blue", stat="count", element="step", alpha=0.5, ax=ax[0])
        sns.histplot(neg[:, 0], bins=bin_edges, label="True", color="orange", stat="count", element="step", alpha=0.5, ax=ax[0])
        sns.histplot(pos[:, 1], bins=bin_edges, label="Predicted", color="blue", stat="count", element="step", alpha=0.5, ax=ax[1])
        sns.histplot(pos[:, 0], bins=bin_edges, label="True", color="orange", stat="count", element="step", alpha=0.5, ax=ax[1])

        ax[1].set_title("Positive Class")
        ax[0].set_title("Negative Class")

        ax[0].set_ylabel("Count")
        ax[1].set_ylabel("")

        ax[1].legend()
        ax[1].yaxis.tick_right()
        plt.suptitle(f'Predicted Probability vs True Class Distribution\n{model_name}')
        plt.tight_layout()
        plt.savefig(pd_path)
        plt.close()



        # Plot predicted probability distribution vs true class distribution
        plt.figure(figsize=(10, 6))
        plt.hist(y_prob[y_true == 0], bins=50, alpha=0.5, label='Predicted Probabilities (True Negatives)')
        plt.hist(y_prob[y_true == 1], bins=50, alpha=0.5, label='Predicted Probabilities (True Positives)')
        plt.axvline(best_threshold, color='r', linestyle='--', label=f'Best Threshold: {best_threshold:.2f}')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Count')
        plt.title(f'Predicted Probability Distribution vs Actual Class Distribution\n{model_name}')
        plt.legend()
        plt.grid(True)
        plt.close()
       
    

        # Save results to CSV
        results_csv = os.path.join(output_path, f'all_model_metrics.csv')
        results_df = pd.DataFrame([metrics])
        if os.path.exists(results_csv):
            pd.concat([pd.read_csv(results_csv), results_df], ignore_index=True).to_csv(results_csv, index=False)
        else:
            results_df.to_csv(results_csv, index=False)
        self.logger.info(f"Metrics saved to {results_csv}")

        # Attempt to log everything to W&B (manager always initializes a run)
        try:
            wandb.log(metrics)
            for key, path in [
                ("roc_curve", roc_path),
                ("pr_curve", pr_path),
                ("probability_distribution", pd_path),
            ]:
                if os.path.exists(path):
                    wandb.log({key: wandb.Image(path)})
            wandb.save(npy_path)
            wandb.save(true_path)
        except Exception:
            self.logger.debug("wandb logging/saving in evaluate() failed")
        
        if model_mgr.sweep_set:
            new_row = pd.DataFrame([{
                'Model': model_name,
                'Experiment': self.experiement_name,
                "n_layers": wandb.config.n_layers,
                "filters": wandb.config.filters,
                "learning_rate": wandb.config.learning_rate,
                "drop_value": wandb.config.drop_value,
                "weight_decay": wandb.config.weight_decay,
                'Accuracy': metrics['Accuracy'],
                'Precision': metrics['Precision'],
                'Recall': metrics['Recall'],
                'F1': metrics['F1'],
                'AUROC': metrics['AUROC'],
                'AP': metrics['AP'],
                'MAE': metrics['MAE'],
                'BCE': metrics['BCE'],
                'Best_Threshold': best_threshold
            }])
            self.hyper_df = pd.concat([self.hyper_df, new_row], ignore_index=True)
            self.hyper_df.to_csv(f"{self.output_dir}/Sweep_results_Model_BCE.csv", index=False)

        return metrics

    def explain(self):
        """ 
        Generate SHAP explanations for the model.
        """
        if self.hazard_model_instance is None:
            self.logger.warning("No trained model found for explanation.")
            return
        
        if self.model_type == 'hazardmodel':
            self.hazard_model_instance.explain_model()
        elif self.model_type == 'baseline':
            self.logger.warning("SHAP explanations not implemented for baseline models.")

    def map(self):
        """ 
        Generate hazard map for the entire region.
        Saves map and plot to output directory.
        """
        # get indices to make map and predict using model
        elevation = np.load(var_paths['elevation'])
        map_shape = elevation.shape

        map_mask = ~np.isnan(elevation)
        map_predictions = self.hazard_model_instance.predict(map_mask)


        # make map with nan values where no data
        full_map = np.full(map_shape, np.nan)
        full_map[map_mask] = map_predictions


        # Save the hazard map npy
        np.save(f'{self.output_dir}/{self.hazard}_hazard_map{ "_downscaled" if downscale else ""}.npy', full_map)

        # Plot and save hazard map
        plot_npy_arrays(
            full_map,
            title=f"{self.hazard.capitalize()} Susceptibility Map{ ' (Downscaled)' if downscale else ''}",
            name=f'{self.hazard} susceptibility',
            type='continuous',
            save_path=f'{self.output_dir}/{self.hazard}_hazard_map{ "_downscaled" if downscale else ""}.png', 
            downsample_factor=1,
            water=True,
            logger=self.logger.info
        )

    def sweep(self):
        """
        Hyperparameter optimization using wandb sweeps
        
        """
        self.logger.info("Starting hyperparameter optimization with wandb")
        

        # Define sweep configuration
        sweep_config = {
            'method': 'bayes',  # Use Bayesian optimization
            'metric': {
                'name': 'val_loss',
                'goal': 'minimize'
            },
            'parameters': {
                'learning_rate': {
                    'distribution': 'log_uniform_values',
                    'min': 0.0005,
                    'max': 0.01
                },
                'weight_decay': {
                    'distribution': 'log_uniform_values',
                    'min': 0.0005,
                    'max': 0.001
                },
                'filters': {
                    'values': [8, 16, 32, 64]
                },
                'n_layers': {
                    'values': [1, 2, 3]
                },
                'drop_value': {
                    'distribution': 'uniform',
                    'min': 0.1,
                    'max': 0.5
                },
            },
        }
        
        # Initialize sweep
        self.hyper_df = pd.DataFrame(columns=['Model', 'Experiment', "n_layers", "filters", "learning_rate", "drop_value", "weight_decay",
                                              'Accuracy', 'Precision', 'Recall', 'F1', 'AUROC', 'AP', 'MAE', 'BCE', 'Best_Threshold'], dtype=float)
        sweep_id = wandb.sweep(
            sweep_config, 
            project=f"{self.hazard}_{self.architecture}_sweep"
        )
         
        # Start the sweep agent
        wandb.agent(sweep_id, self.main, count=10)  # Run 50 trials
        self.logger.info(f"Hyperparameter sweep completed with ID: {sweep_id}")

    def main(self, config: dict | None = None):
        """
        Train/evaluate once. Used directly or by W&B sweeps.
        """
        # merge defaults + overrides
        hparams = self._default_hparams()

        self.logger.info(f"FILTERS: {hparams['filters']}")
        self.logger.info(f"LAYERS: {hparams['n_layers']}")
        self.logger.info(f"hparams: {hparams}")
        self.logger.info(f"LEARNING RATE: {hparams['learning_rate']}")

        if os.path.exists(f"{self.output_dir}/Sweep_results_Model_BCE.csv"):
            self.logger.info("PREVIOUS SWEEP FOUND, PULL BEST HPARAMS")
            df = pd.read_csv(f"{self.output_dir}/Sweep_results_Model_BCE.csv")
            row = df.sort_values(by="BCE", ascending=True).iloc[0]  # "val_loss"
            hparams['filters'] = int(row['filters'])
            hparams['n_layers'] = int(row['n_layers'])
            hparams["learning_rate"] = np.round(row["learning_rate"], 5)
            hparams["weight_decay"] = np.round(row["weight_decay"], 5)
            hparams["drop_value"] = np.round(row["drop_value"], 2)

        self.logger.info(f"FILTERS: {hparams['filters']}")
        self.logger.info(f"LAYERS: {hparams['n_layers']}")
        self.logger.info(f"LEARNING RATE: {hparams['learning_rate']}")

        self.logger.info(f"CONFIG: {config}")
        if config:
            hparams.update(config)
        # init wandb (safe for both normal + sweep)
        try:
            wandb.init(
                project=self.hazard,
                config=hparams,
                name=f"{self.architecture}_{self.experiement_name}",
            )
        except Exception:
            self.logger.debug("wandb.init() failed or no active run")

        # pull effective config
        if self.sweep_set:
            self.logger.info("THIS SHOULD BE A SWEEP RUN")
            cfg = dict(wandb.config)
        else:
            self.logger.info("HPARAMS ARE BY DEFAULT OR UPDATED IF SUCCESSFULLY PULLED")
            cfg = hparams

        # build and configure model
        self.hazard_model_instance = self.build_model()
        for k, v in cfg.items():
            if hasattr(self.hazard_model_instance, k):
                setattr(self.hazard_model_instance, k, v)

        # train, predict, evaluate
        y_true, y_prob = self.hazard_model_instance.main()
        self.evaluate(y_true, y_prob)

        # try:
        #     wandb.finish()
        # except Exception:
        #     self.logger.debug("wandb.finish() failed")

    def _default_hparams(self):
        """Default model/training hyperparameters only."""
        return {
            "learning_rate": 2e-3,
            "weight_decay": 5e-5,
            "filters": 64,
            "n_layers": 3,
            "drop_value": 0.3,
            "kernel_size": 3,
            "pool_size": 2,
            "n_nodes": 128,
            "dropout": True,
        }

def validate_inputs(args):
    valid_hazards = label_paths.keys()
    valid_architectures = ['SimpleCNN', 'SpatialAttentionCNN', 'MLP', 'LR', 'RF', 'CNN', 'CNNatt', 'CNN_GAP', 'CNN_GAPatt']

    if args.hazard not in valid_hazards:
        raise ValueError(f"Invalid hazard type: {args.hazard}. Choose from {valid_hazards}.")
    
    if args.architecture not in valid_architectures:
        raise ValueError(f"Invalid architecture: {args.architecture}. Choose from {valid_architectures}.")

    if args.batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")
    
    if args.patch_size <= 0 or args.patch_size % 2 == 0:
        raise ValueError("Patch size must be a positive odd integer.")
    
    if args.epochs <= 0:
        raise ValueError("Number of epochs must be a positive integer.")

    if args.sweep and args.architecture not in ['SimpleCNN', 'SpatialAttentionCNN', 'MLP', 'CNN_GAP','CNN', 'CNNatt', 'CNN_GAPatt']:
        raise ValueError("Hyperparameter sweep is only supported for torch architectures.")

    if args.explain and args.architecture not in ['SimpleCNN', 'SpatialAttentionCNN', 'MLP', 'CNN', 'CNN_GAP', 'CNNatt', 'CNN_GAPatt']:
        raise ValueError("SHAP explanations are only supported for torch architectures.")

    return True


if __name__ == "__main__":
  # Example configuration

    # use argparser 
    parser = argparse.ArgumentParser(description='Hazard Mapper Configuration')
    parser.add_argument('-n', '--name', type=str, default='HazardMapper', help='Name of the experiment')
    parser.add_argument('-z', '--hazard', type=str, default='landslide', help='Hazard type (wildfire, landslide or flood)')
    parser.add_argument('-b', '--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('-p', '--patch_size', type=int, default=5, help='Patch size for model input')
    parser.add_argument('-a', '--architecture', type=str, default='CNN', help='Model architecture (LR, RF, MLP, CNN, SimpleCNN, SpatialAttentionCNN, CNNatt, CNN_GAP, CNN_GAPatt)')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--sweep', action='store_true', default=False,  help='Enable hyperparameter optimization')
    parser.add_argument('--map', action='store_true', default=False, help='Create hazard map after training')
    parser.add_argument('--explain', action='store_true', default=False, help='Compute SHAP values for model explanations')
    parser.add_argument('--downscale', action='store_true', default=False, help='Use downscaled maps for testing')
    args = parser.parse_args()
    validate_inputs(args)

    hazard = args.hazard
    batch_size = args.batch_size
    patch_size = args.patch_size
    architecture = args.architecture
    epoch = args.epochs
    sweep = args.sweep
    experiement_name = args.name
    make_map = args.map
    explain = args.explain
    downscale = args.downscale
    


    if downscale:
        print("Using downscaled maps for testing")
        var_paths = var_paths_downscaled
        label_paths = label_paths_downscaled
        partition_paths = partition_paths_downscaled


    # Initialize the model manager
    model_mgr = ModelMgr(
        batch_size=batch_size,
        patch_size=patch_size,
        architecture=architecture,
        epoch=epoch,
        hazard=hazard,
        experiement_name=experiement_name,
    )
    # Run hyperparameter sweep or single training/evaluation
    if sweep:
        model_mgr.sweep_set = True
        model_mgr.sweep()
    else:
        model_mgr.sweep_set = False
        model_mgr.main()
    # If specified, create a hazard map
    if make_map:
        model_mgr.map()
    # If specified, compute SHAP explanations
    if explain:
        model_mgr.explain()

    # Temporary because wandb sometimes hangs on exit
    try:
        wandb.finish()
    except Exception:
        model_mgr.logger.debug("wandb.finish() failed or no run")    
    os._exit(0)
    