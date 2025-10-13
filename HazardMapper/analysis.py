"""HazardMapper - Analysis Module
=========================
This module provides functionality for exploratory data analysis (EDA) and results evaluation of hazard models.
It includes methods to compute hazard statistics, visualize distributions, and evaluate model performance.

This module also generate the results for the thesis manuscript, including LaTeX tables and figures for the hazards in Europe.
""" 

import os
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, mean_absolute_error, precision_recall_curve, roc_auc_score, average_precision_score, roc_curve
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import argparse
import jenkspy
import time
import json
import seaborn as sns
from HazardMapper.utils import plot_npy_arrays, normalize_map

class EDA:
    def __init__(self):
        # Load hazard inventory and region map
        self.hazards = ['landslide', 'flood', 'wildfire']
        self.hazard_inventories = [np.load(f"Input/Europe/npy_arrays/masked_{hazard}_Europe.npy").flatten() for hazard in self.hazards]
        self.region_map = np.load("Input/Europe/partition_map/sub_countries_rasterized.npy").flatten()
        self.valid_mask = ~np.isnan(np.load("Input/Europe/npy_arrays/masked_elevation_Europe.npy")).flatten()


        # Apply valid mask to region map and hazard inventory
        self.region_map = self.region_map[self.valid_mask]
        self.hazard_inventories = [hazard_inventory[self.valid_mask] for hazard_inventory in self.hazard_inventories]

        # Create output directory if it doesn't exist
        self.output_dir = f'Output/Europe/eda'
        os.makedirs(self.output_dir, exist_ok=True)

        self.hazard_stats = None
        self.region_stats = None

        # self.load_stats()

    def store_stats(self):
        # Store region stats in a JSON file
        region_stats_path = f'{self.output_dir}/region_stats.json'
        with open(region_stats_path, 'w') as f:
            json.dump(self.region_stats, f, indent=4)
        print(f'Region stats stored at {region_stats_path}')
    
    def load_stats(self):
        # Load region stats from a JSON file
        region_stats_path = f'{self.output_dir}/region_stats.json'
        if os.path.exists(region_stats_path):
            with open(region_stats_path, 'r') as f:
                self.region_stats = json.load(f)
            print(f'Region stats loaded from {region_stats_path}')
        else:
            print(f'No region stats found at {region_stats_path}, please run get_region_stats() first.')

    def get_hazard_stats(self):
        # Initialize hazard stats dictionary
        self.hazard_stats = {
            'hazard': [],
            'n_hazard_pixels': [],
            'n_pixels': [],
            'hazard_coverage': [],
            'avg_hazard_count': [],
            'min_hazard_count': [],
            'max_hazard_count': [],
            'std_hazard_count': [],
            'hazard_counts': [],
        }

        # Calculate total hazard counts, total hazard pixels, and hazard percentage
        for hazard_inventory, hazard in zip(self.hazard_inventories, self.hazards):
            print(f'Calculating hazard stats for {hazard} hazard...')

            n_hazard_pixels = np.sum(hazard_inventory > 0)
            n_total_pixels = np.sum(self.region_map > 0)
            hazard_percentage = round((n_hazard_pixels / n_total_pixels) * 100, 2) if n_total_pixels > 0 else 0
            self.hazard_stats['hazard'].append(hazard)
            self.hazard_stats['n_hazard_pixels'].append(n_hazard_pixels)
            self.hazard_stats['n_pixels'].append(n_total_pixels)
            self.hazard_stats['hazard_coverage'].append(hazard_percentage)
        print(f'Hazard stats for {hazard} hazard:')
        print(f'  - Total hazard pixels: {self.hazard_stats["n_hazard_pixels"]}')
        print(f'  - Total pixels: {self.hazard_stats["n_pixels"]}')
        print(f'  - Hazard coverage: {self.hazard_stats["hazard_coverage"]}%')

        # Get region-wise hazard statistics
        self.get_region_stats()

    def get_region_stats(self):
        # Initialize region stats dictionary
        self.region_stats = {
            'region_ID': [],
            'n_hazard_pixels': [],
            'n_pixel': [],
            'hazard_coverage': [],
        }

        # Calculate region-wise hazard statistics for eah hazard
        for hazard_inventory, hazard in zip(self.hazard_inventories, self.hazards):
            print(f'Calculating region stats for {hazard} hazard...')

            unique_regions = np.unique(self.region_map)
            for region in unique_regions:
                if region == 0:
                    continue
                region_mask = self.region_map == region
                n_hazard_pixels = np.sum(hazard_inventory[region_mask] > 0)
                n_total_pixels = np.sum(region_mask)
                hazard_coverage = (n_hazard_pixels / n_total_pixels) * 100 if n_total_pixels > 0 else 0
                self.region_stats['region_ID'].append(region)
                self.region_stats['n_hazard_pixels'].append(n_hazard_pixels)
                self.region_stats['n_pixel'].append(n_total_pixels)
                self.region_stats['hazard_coverage'].append(hazard_coverage)

            # Calculate average, min, max, and std of hazard counts for the hazard
            avg_hazard_count = np.mean(self.region_stats['n_hazard_pixels'])
            min_hazard_count = np.min(self.region_stats['n_hazard_pixels'])
            max_hazard_count = np.max(self.region_stats['n_hazard_pixels'])
            std_hazard_count = np.std(self.region_stats['n_hazard_pixels'])
        

            self.hazard_stats['avg_hazard_count'].append(avg_hazard_count)
            self.hazard_stats['min_hazard_count'].append(min_hazard_count)
            self.hazard_stats['max_hazard_count'].append(max_hazard_count)
            self.hazard_stats['std_hazard_count'].append(std_hazard_count)

            self.hazard_stats['hazard_counts'].append(self.region_stats['hazard_coverage'])

            print(f'Region stats for {hazard} hazard:')
            print(f'  - Average hazard counts: {avg_hazard_count}')
            print(f'  - Min hazard counts: {min_hazard_count}')
            print(f'  - Max hazard counts: {max_hazard_count}')
            print(f'  - Std hazard counts: {std_hazard_count}')
       
    def make_latex_table(self):
        # Create a LaTeX table for the hazard with total coverage, total hazard pixels, averege region coverage, min and max region coverage, std 
        print(f'Creating LaTeX table for hazards...')
        table = "\\begin{table}[h!]\n"
        table += "    \\centering\n"
        table += "    \\begin{tabularx}{\\textwidth}{@{}lXXXXXX@{}}\n"
        table += "    \\toprule\n"
        table += "    \\textbf{Hazard} & \\textbf{Total} & \\textbf{Min} & \\textbf{Avg} & \\textbf{Max} & \\textbf{Std} \\\\\n"
        table += "    \\midrule\n"
        for hazard, n_hazard_pixels, min_hazard_count, max_hazard_count, avg_hazard_count, std_hazard_count in zip(self.hazard_stats['hazard'],
                                                                        self.hazard_stats['n_hazard_pixels'],
                                                                        self.hazard_stats['min_hazard_count'],
                                                                        self.hazard_stats['avg_hazard_count'],
                                                                        self.hazard_stats['max_hazard_count'],
                                                                        self.hazard_stats['std_hazard_count']):
            # Format the statistics to two decimal places
            avg_hazard_count = f"{avg_hazard_count:.2f}"
            std_hazard_count = f"{std_hazard_count:.2f}"                                                           
            table += f"    {hazard} & {n_hazard_pixels} & {min_hazard_count} & {avg_hazard_count} & {max_hazard_count} & {std_hazard_count}\\% \\\\\n"
        table += "    \\bottomrule\n"
        table += "    \\end{tabularx}\n"
        table += f"    \\caption{"Total number of susceptible pixels per hazard in the area of of interest. The counts are also inspected per NUTS-2 region, with the minimum, average, maximum and standard deviation also included for each hazard. These statistics along with the figure XX visualize the diversity and bias in the hazard inventories.   "}\n"
        table += "    \\label{tab:hazard_stats}\n"
        table += "\\end{table}\n"
        with open(f'{self.output_dir}/hazard_stats.tex', 'w') as f:
            f.write(table)
        print(f'LaTeX table for hazards created at {self.output_dir}/hazard_stats.tex')

    def plot_hazard_distribution(self):
       # Plot the distribution of region-wise hazard coverage for each hazard
        print(f'Plotting regional hazard count distribution...')
        fig, axs = plt.subplots(1, len(self.hazards), figsize=(10, 6))
        for i, hazard in enumerate(self.hazards):
            hazard_counts = self.hazard_stats['hazard_counts'][i]
            sns.histplot(hazard_counts, bins=20, kde=True, ax=axs[i])
            axs[i].set_title(f'{hazard.capitalize()}')
            axs[i].set_xlabel('Hazard Counts')
            axs[i].set_ylabel('Frequency')
        plt.suptitle('Regional Hazard Count Distribution')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/hazard_distribution.png')
        plt.close()

class Results:
    def __init__(self, hazard):
        self.hazard = hazard
        y_true_file = f'Output/Europe/{hazard}/evaluation/true_labels.npy'
        self.y_true = np.load(y_true_file)
        self.y_pred = []
        self.y_prob = []
        self.models = []
        self.scores = []
        self.colors = []  # Use a colormap for consistent colors
       
        self.figsize = (6, 4)
        self.cmap = 'viridis'  # Colormap for model colors

        self.output_dir = f'Output/Europe/{hazard}/results'
        os.makedirs(self.output_dir, exist_ok=True)

    def load_predictions(self):
        print(f'Loading predictions for {self.hazard} hazard...')
        dir = f'Output/Europe/{self.hazard}/evaluation'

        # List all npy files in the directory
        files = [f for f in os.listdir(dir) if f.endswith('.npy')]
        
        # Define the desired order of models
        desired_order = ['LR', 'RF', 'MLP', 'SimpleCNN', 'SpatialAttentionCNN']

        # File name structure: hazard_model_prediction.npy
        for file in files:
            parts = file.split('_')
            if (parts[-1] == 'predictions.npy') & (parts[-2] in desired_order):
                model_name = parts[1]
                y_prob = np.load(os.path.join(dir, file))

                self.models.append(model_name)
                print(f'Loaded predictions from {file} for model {model_name}, with length {len(y_prob)}')
                self.y_prob.append(y_prob)
        # Sort models by name for consistent ordering
        # Define your fixed desired order.

        # Combine the two lists
        combined = list(zip(self.models, self.y_prob))

        # Sort using the index in desired_order (if a model is missing, give it a high index)
        combined.sort(key=lambda x: desired_order.index(x[0]) if x[0] in desired_order else 999)

        # Unzip back into self.models and self.y_prob
        self.models, self.y_prob = map(list, zip(*combined))

        # Generate a color for the model using a colormap
        cmap = plt.get_cmap(self.cmap, len(self.models))
        self.colors = [cmap(i) for i in range(cmap.N)]
        
    def get_scores(self):

        self.load_predictions()


        print(f'Calculating scores for {self.hazard} hazard...')
        for y_prob, model_name in zip(self.y_prob, self.models):
            # Optimize threshold based on best F1
            precision_curve, recall_curve, thresholds = precision_recall_curve(self.y_true, y_prob)
            f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-10)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5


            # Get predictions with best threshold
            y_pred =(y_prob >= best_threshold).astype(int)

            # Compute metrics
            metrics = {
                'Model': model_name,
                'F1': f1_score(self.y_true, y_pred, zero_division=0),
                'AUC': roc_auc_score(self.y_true, y_prob),
                'AP': average_precision_score(self.y_true, y_prob),
                'MAE': mean_absolute_error(self.y_true, y_prob),
                'Best_Threshold': best_threshold
            }
            self.scores.append(metrics)


    def get_scoresV2(self, n_bootstraps=100, ci=95, random_seed=42):
        """
        Computes point estimates, 95% bootstrap CIs for F1, AUC, and MAE,
        and performs significance tests comparing CNN to all other models.
        Stores results in instance attributes for later use.
        """
        # Load predictions and true labels
        self.load_predictions()
        y_true = self.y_true
        y_prob_list = self.y_prob
        model_names = self.models

        # Determine optimal thresholds and point metrics
        thresholds = {}
        self.point_metrics = {}
        for y_prob, name in zip(y_prob_list, model_names):
            precision, recall, thr = precision_recall_curve(y_true, y_prob)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.nanargmax(f1_scores)
            thresh = thr[best_idx] if best_idx < len(thr) else 0.5
            thresholds[name] = thresh
            y_pred = (y_prob >= thresh).astype(int)

            self.point_metrics[name] = {
                'f1': f1_score(y_true, y_pred),
                'auc': roc_auc_score(y_true, y_prob),
                'mae': mean_absolute_error(y_true, y_prob)
            }

        # Bootstrap for CIs
        rng = np.random.RandomState(random_seed)
        n_samples = len(y_true)
        boot_metrics = {name: {'f1': [], 'auc': [], 'mae': []} for name in model_names}

        for _ in range(n_bootstraps):
            idx = rng.randint(0, n_samples, n_samples)
            y_true_bs = y_true[idx]
            for y_prob, name in zip(y_prob_list, model_names):
                y_prob_bs = y_prob[idx]
                y_pred_bs = (y_prob_bs >= thresholds[name]).astype(int)
                boot_metrics[name]['f1'].append(f1_score(y_true_bs, y_pred_bs))
                boot_metrics[name]['auc'].append(roc_auc_score(y_true_bs, y_prob_bs))
                boot_metrics[name]['mae'].append(mean_absolute_error(y_true_bs, y_prob_bs))

        # Compute CI bounds
        alpha = (100 - ci) / 2
        self.ci_metrics = {}
        for name in model_names:
            f1_ci = np.percentile(boot_metrics[name]['f1'], [alpha, 100 - alpha])
            auc_ci = np.percentile(boot_metrics[name]['auc'], [alpha, 100 - alpha])
            mae_ci = np.percentile(boot_metrics[name]['mae'], [alpha, 100 - alpha])
            self.ci_metrics[name] = {
                'f1': (f1_ci[0], f1_ci[1]),
                'auc': (auc_ci[0], auc_ci[1]),
                'mae': (mae_ci[0], mae_ci[1])
            }

        # Significance tests using paired Wilcoxon on bootstrap replicates
        cnn_candidates = [m for m in model_names if 'cnn' in m.lower()]
        self.sig_tests = {}
        if cnn_candidates:
            cnn_name = cnn_candidates[0]
            for name in model_names:
                if name == cnn_name:
                    continue
                self.sig_tests[name] = {}
                for metric in ('f1', 'auc', 'mae'):
                    # arrays of per-bootstrap scores
                    cnn_scores = np.array(boot_metrics[cnn_name][metric])
                    other_scores = np.array(boot_metrics[name][metric])
                    # paired Wilcoxon, two-sided
                    stat, p = wilcoxon(cnn_scores, other_scores, zero_method='wilcox', alternative='two-sided')
                    self.sig_tests[name][metric] = p

    
    
    def make_latex_tableV3(self):
        print(f'Creating LaTeX table for {self.hazard} hazard...')
        table = "\\begin{table*}[h!]\n"
        table += "    \\centering\n"
        table += "    \\begin{tabularx}{0.75\\textwidth}{@{}lXXXX@{}}\n"
        table += "    \\toprule\n"
        table += "    \\textbf{Model} & \\textbf{F1} & \\textbf{AUC} & \\textbf{MAE} \\\\\n"
        table += "    \\midrule\n"
        
        for m in self.models:
            pm = self.point_metrics[m]
            low_f, high_f = self.ci_metrics[m]['f1']
            low_auc, high_auc = self.ci_metrics[m]['auc']
            low_mae, high_mae = self.ci_metrics[m]['mae']
            
            table += (
                f"    {m} & "
                f"{pm['f1']:.3f} [{low_f:.3f}, {high_f:.3f}] & "
                f"{pm['auc']:.3f} [{low_auc:.3f}, {high_auc:.3f}] & "
                f"{pm['mae']:.3f} [{low_mae:.3f}, {high_mae:.3f}] \\\\\n"
            )

        table += "    \\bottomrule\n"
        table += "    \\end{tabularx}\n"
        table += f"    \\caption{{Evaluation metrics for models on {self.hazard} hazard.}}\n"
        table += "    \\label{tab:evaluation_metrics}\n"
        table += "\\end{table*}\n"

        with open(f'{self.output_dir}/metric_table.tex', 'w') as f:
            f.write(table)

    def make_pr_curve(self):

        print(f'Creating Precision-Recall curve for {self.hazard} hazard...')
        plt.figure(figsize=self.figsize)
        for y_prob, model_name, color in zip(self.y_prob, self.models, self.colors):
            precision_curve, recall_curve, _ = precision_recall_curve(self.y_true, y_prob)
            ap = average_precision_score(self.y_true, y_prob)
            plt.plot(recall_curve, precision_curve, 
                     label=f'{model_name} $AP={ap:.2f}$', 
                     color=color)


        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.title(f'{self.hazard.capitalize()} Precision-Recall Curve')
        plt.legend(loc='best', fontsize=10, frameon=True)
        plt.grid()
        plt.savefig(f'{self.output_dir}/pr_curve.pdf')
        plt.close()
    
    def make_roc_curve(self):
        """
        Creates a ROC curve for the models and saves it as an image.
        """
        print(f'Creating ROC curve for {self.hazard} hazard...')
        plt.figure(figsize=self.figsize)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing $AUC=0.5$')
        for y_prob, model_name, color in zip(self.y_prob, self.models, self.colors):
            fpr, tpr, _ = roc_curve(self.y_true, y_prob)
            roc_auc = roc_auc_score(self.y_true, y_prob)
            plt.plot(fpr, tpr, 
                     label=f'{model_name} $AUC={roc_auc:.2f}$', 
                     color=color)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{self.hazard.capitalize()} ROC Curve')
        plt.legend(loc='best', fontsize=10, frameon=True)
        plt.grid()
        plt.savefig(f'{self.output_dir}/roc_curve.pdf')
        plt.close()
    
    def make_probability_histogram(self):
        """
        Creates a histogram of the predicted probabilities for each model and saves it as an image.
        """
        print(f'Creating probability histogram for {self.hazard} hazard...')
        plt.figure(figsize=(10, 6))
        for y_prob, model_name in zip(self.y_prob, self.models):
            sns.histplot(y_prob, bins=50, kde=True, label=model_name, stat='density', alpha=0.5)

        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title(f'Predicted Probability Histogram for {self.hazard} Hazard')
        plt.legend()
        plt.grid()
        plt.savefig(f'{self.output_dir}/probability_histogram.png')
        plt.close()
    
    def make_latex_table(self):
        print(f'Creating LaTeX table for {self.hazard} hazard...')
        table = "\\begin{table}[h!]\n"
        table += "    \\centering\n"
        table += "    \\begin{tabularx}{0.45\\textwidth}{@{}lXXXXX@{}}\n"
        table += "    \\toprule\n"
        table += "    \\textbf{Model} & \\textbf{F1} & \\textbf{AUC} & \\textbf{MAE} \\\\\n"
        table += "    \\midrule\n"
        
        for score in self.scores:
            table += f"    {score['Model']} & {score['F1']:.3f} & {score['AUC']:.3f}  & {score['MAE']:.3f} \\\\\n"

        table += "    \\bottomrule\n"
        table += "    \\end{tabularx}\n"
        table += f"    \\caption{{Evaluation metrics for models on {self.hazard} hazard.}}\n"
        table += "    \\label{tab:evaluation_metrics}\n"
        table += "\\end{table}\n"
        
        with open(f'{self.output_dir}/metric_table.tex', 'w') as f:
            f.write(table)

    def plot_output_distribution(self):
        """
        Plots the distribution of the test set comparing the hazard map and the true labels.

        """

        print(f'Plotting output distribution for {self.hazard} hazard...')
        bins = np.linspace(0, 1, 11)  # Adjust bins as needed
        plt.figure(figsize=(10, 6))
        sns.histplot(self.y_true, bins=bins, kde=True, label='True Labels', color='blue', alpha=0.5)
        sns.histplot(self.y_prob, bins=bins, kde=True, label='Predictions', color='orange', alpha=0.5)
        plt.xlabel('Susceptibility Score')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {self.hazard} Model Output vs True Labels')
        plt.legend()
        plt.grid()
        plt.savefig(f'{self.output_dir}/output_distribution.png')
        plt.close()
    
    def plot_calibration_curve(self):
        """
        Plots the calibration curve for the predicted probabilities.
        Automatically chooses the best location for the legend.
        """

        fig, ax = plt.subplots(figsize=self.figsize)

        for model, y_prob, color in zip(self.models, self.y_prob, self.colors):
            brier_score = np.mean((self.y_true - y_prob) ** 2)
            print(f'Plotting calibration curve for {model} with Brier score: {brier_score:.2f}')
            CalibrationDisplay.from_predictions(
                self.y_true,
                y_prob,
                n_bins=10,
                name=f'{model} $BS={brier_score:.2f}$',
                strategy='uniform',
                color=color,
                ax=ax
            )

        ax.set_ylabel('Fraction of Positives')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_title(f'{self.hazard.capitalize()} Calibration Curves')
        ax.grid(True)
        ax.legend(loc='best', fontsize=10, frameon=True)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/calibration_curve.pdf')
        plt.close()
        print(f'Calibration curves saved to {self.output_dir}/calibration_curve.pdf')


    def bin_map(self, model="SpatialAttentionCNN"):
        """
        Applies Jenks Natural Breaks classification to a .npy array of continuous values,
        and computes the Goodness of Variance Fit (GVF) for the classification.

        Returns:
        - binned_array (np.ndarray): Array with the same shape as input, but values replaced by bin indices (0 to n_classes - 1)
        - breaks (List[float]): List of break thresholds used for binning
        - gvf (float): Goodness of Variance Fit for the Jenks classification
        """
        # load your map
        map_path = f'Output/Europe/{self.hazard}/hazard_map/{model}_{self.hazard}_hazard_map.npy'
        hazard_map = np.load(map_path)

        # sample for performance
        flat = hazard_map[~np.isnan(hazard_map)].ravel()
        np.random.shuffle(flat)
        flat = flat[:100_000]

        n_classes = 5
        print(f'Calculating Jenks breaks for {self.hazard} with {n_classes} classes...')
        breaks = jenkspy.jenks_breaks(flat, n_classes=n_classes)
        print(f'Jenks breaks: {breaks}')

        # --- compute GVF ---
        # total variance around the global mean
        mean_all = flat.mean()
        SDAM = np.sum((flat - mean_all)**2)

        # within‐class variance
        # digitize returns classes 0..n_classes-1
        cls_idx = np.digitize(flat, bins=breaks[1:-1], right=False)
        SDCM = 0.0
        for j in range(n_classes):
            members = flat[cls_idx == j]
            if members.size:
                SDCM += np.sum((members - members.mean())**2)

        gvf = (SDAM - SDCM) / SDAM
        print(f'Goodness of Variance Fit (GVF): {gvf:.4f}')

        # Save breaks to CSV (unchanged)
        breaks_path = f'Output/Europe/{self.hazard}/hazard_map/{model}_jenks_breaks.csv'
        if os.path.exists(breaks_path):
            df = pd.read_csv(breaks_path)
            df = df[df['Model'] != model]
            new_row = pd.DataFrame({
                "Hazard":[self.hazard],
                "Model":[model],
                "Number of Classes":[n_classes],
                "Breaks":[','.join(map(str, breaks))],
                "GVF":[gvf]
            })
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = pd.DataFrame({
                "Hazard":[self.hazard],
                "Model":[model],
                "Number of Classes":[n_classes],
                "Breaks":[','.join(map(str, breaks))],
                "GVF":[gvf]
            })
        df.to_csv(breaks_path, index=False)

        # digitize full map
        binned = np.digitize(hazard_map, bins=breaks[1:-1], right=False).astype(float)
        binned[np.isnan(hazard_map)] = np.nan

        # save & plot
        out_np = f'Output/Europe/{self.hazard}/hazard_map/{model}_{self.hazard}_binned_hazard_map.npy'
        np.save(out_np, binned)
        plot_npy_arrays(
            binned,
            name='Susceptibility',
            type='bins',
            title=f'{self.hazard} Susceptibility Map',
            save_path=f'Output/Europe/{self.hazard}/hazard_map/{model}_{self.hazard}_binned_hazard_map.png'
        )

        return binned, breaks, gvf
        
class MHMap:
    def __init__(self):

        self.hazard_paths = [
        "Output/Europe/landslide/hazard_map/SimpleCNN_landslide_hazard_map.npy",
        "Output/Europe/wildfire/hazard_map/SimpleCNN_wildfire_hazard_map.npy",
        "Output/Europe/flood/hazard_map/SimpleCNN_flood_hazard_map.npy",
        "Input/Europe/npy_arrays/masked_heatwave_Europe.npy",
        "Input/Europe/npy_arrays/masked_pga_Europe.npy",
        "Input/Europe/npy_arrays/masked_drought_Europe.npy",
        "Input/Europe/npy_arrays/masked_extreme_wind_Europe.npy",
        "Input/Europe/npy_arrays/masked_volcano_Europe.npy",
        ]

        self.hazard_maps = [np.load(path) for path in self.hazard_paths if os.path.exists(path)]

        mh_map = self.make_multi_hazard_map(self.hazard_maps)
        self.bin_multi_hazard_map(mh_map)

    
    def make_multi_hazard_map(self, hazard_maps):
        """
        Create a multi-hazard map from individual hazard maps.
        
        Args:
            hazard_maps (list of np.ndarray): List of hazard maps to combine.
            
        Returns:
            np.ndarray: Combined multi-hazard map.
        """
        multi_hazard_map = np.zeros_like(hazard_maps[0], dtype=np.float32)
        
        for i, hazard_map in enumerate(hazard_maps):
            normalized_map = normalize_map(hazard_map)
            # Combine the maps by averaging
            multi_hazard_map += normalized_map 
        
        return multi_hazard_map
    
    def bin_multi_hazard_map(self, hazard_map):
        """
        Applies Jenks Natural Breaks classification to a .npy array of continuous values,
        and computes the Goodness of Variance Fit (GVF) for the classification.

        Returns:
        - binned_array (np.ndarray): Array with the same shape as input, but values replaced by bin indices (0 to n_classes - 1)
        - breaks (List[float]): List of break thresholds used for binning
        - gvf (float): Goodness of Variance Fit for the Jenks classification
        """
     
        hazard = 'multi_hazard'

        # sample for performance
        flat = hazard_map[~np.isnan(hazard_map)].ravel()
        np.random.shuffle(flat)
        flat = flat[:100_000]

        n_classes = 5
        print(f'Calculating Jenks breaks for {hazard} with {n_classes} classes...')
        breaks = jenkspy.jenks_breaks(flat, n_classes=n_classes)
        print(f'Jenks breaks: {breaks}')

        # --- compute GVF ---
        # total variance around the global mean
        mean_all = flat.mean()
        SDAM = np.sum((flat - mean_all)**2)

        # within‐class variance
        # digitize returns classes 0..n_classes-1
        cls_idx = np.digitize(flat, bins=breaks[1:-1], right=False)
        SDCM = 0.0
        for j in range(n_classes):
            members = flat[cls_idx == j]
            if members.size:
                SDCM += np.sum((members - members.mean())**2)

        gvf = (SDAM - SDCM) / SDAM
        print(f'Goodness of Variance Fit (GVF): {gvf:.4f}')


        # digitize full map
        binned = np.digitize(hazard_map, bins=breaks[1:-1], right=False).astype(float)
        binned[np.isnan(hazard_map)] = np.nan

        # save & plot
        out_np = f'Output/Europe/{hazard}/hazard_map/{hazard}_binned_hazard_map.npy'
        np.save(out_np, binned)
        plot_npy_arrays(
            binned,
            name='Susceptibility',
            type='bins',
            water=True,
            title=f'Multi Hazard Susceptibility Map',
            # save_path=f'Output/Europe/{hazard}/hazard_map/{hazard}_binned_hazard_map.png'
            save_path=f'Output/Europe/cover.png'

        )

        return binned, breaks, gvf
    


if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description='Evaluate hazard models')
    argparse.add_argument('-z', '--hazard', type=str, help='Hazard type (wildfire, landslide, flood)')
    argparse.add_argument('-e', '--eda', action='store_true', help='Run exploratory data analysis')
    argparse.add_argument('-r', '--results', action='store_true', help='Run results analysis')
    argparse.add_argument('-m', '--model', type=str ,required=False, help='Model name for results analysis (default: SpatialAttentionCNN)', default='SpatialAttentionCNN')
    argparse.add_argument('-mh', '--mhmap', action='store_true', help='Run multi-hazard map analysis')
    args = argparse.parse_args()

    plt.style.use('bauhaus_light')


    if args.eda:
        eda = EDA()
        eda.get_hazard_stats()
        # eda.make_latex_table()
        eda.plot_hazard_distribution()

    if args.results:
        results = Results(args.hazard)
        results.get_scoresV2()
        # results.make_latex_tableV3()
        results.make_pr_curve()
        results.make_roc_curve()
        results.make_latex_table()
        results.plot_calibration_curve()
        # results.bin_map(model=args.model)

    if args.mhmap:
        mhmap = MHMap()
      

