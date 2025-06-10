import os
import json
import torch
import multiprocessing
from functools import partial

from PDCM.pdcm_parallel import PDCMTrainer


def refit_model_if_needed(subject_id, region_index, exp, base_dir, max_retries=3, n_trials=100, num_epochs=1):
    """
    Refit a PDCM model for a specific subject, ROI, and experiment condition
    if the current model's MSE is not better than the baseline.

    Parameters:
        subject_id (int): ID of the subject (e.g., 1, 2, 3...)
        region_index (int): Index of the brain region (ROI)
        exp (str): Experiment condition (e.g., "PLCB", "LSD")
        base_dir (str): Base directory where model outputs are stored
        max_retries (int): Maximum number of re-optimization attempts
        n_trials (int): Number of Optuna trials for hyperparameter tuning
        num_epochs (int): Number of epochs to train during each retry
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Construct paths for subject and experiment files
    sub_folder = f"sub-{subject_id:03d}"
    prefix = f"roi-{region_index}-{exp}"
    root = os.path.join(base_dir, sub_folder)

    result_path = os.path.join(root, f"{prefix}_results.json")
    metric_path = os.path.join(root, f"{prefix}_metrics.json")
    baseline_path = os.path.join(root, f"{prefix}_baseline_metrics.json")

    # Check that all necessary files exist
    if not os.path.exists(result_path) or not os.path.exists(metric_path) or not os.path.exists(baseline_path):
        print(f" Missing files for Subject {subject_id}, ROI {region_index}, {exp}")
        return

    # Load baseline MSE for comparison
    try:
        with open(baseline_path, 'r') as f:
            baseline_metrics = json.load(f)
            baseline_mse = baseline_metrics["normalised_csd_metrics"]["mse"]
    except Exception as e:
        print(f" Could not load baseline metrics for {prefix}: {e}")
        return

    # Load existing results and metrics
    try:
        with open(metric_path, 'r') as f:
            best_metrics = json.load(f)
            best_mse = best_metrics["normalised_csd_metrics"]["mse"]
        with open(result_path, 'r') as f:
            best_result = json.load(f)
    except Exception as e:
        print(f" Could not load current metrics for {prefix}: {e}")
        best_mse = float('inf')  # Default if current model fails to load
        best_metrics = {}
        best_result = {}

    # Skip retraining if current model is already better than baseline
    if best_mse < baseline_mse:
        print(f" Already good: Subject={subject_id}, ROI={region_index}, {exp} | MSE {best_mse:.4f} < {baseline_mse:.4f}")
        return

    print(f" Refitting: Subject={subject_id}, ROI={region_index}, {exp} | MSE {best_mse:.4f} >= Baseline {baseline_mse:.4f}")
    attempt = 0

    # Retry optimization if baseline is better
    while attempt < max_retries:
        try:
            trainer = PDCMTrainer(subject_id, region_index, exp, device)

            # Perform hyperparameter optimization using Optuna
            best_params = trainer.run_optuna(n_trials=n_trials)

            # Train model using best parameters
            trainer.train(best_params, num_epochs=num_epochs)
            trainer.save_results()

            # Evaluate and compare new MSE to current best
            with open(metric_path, 'r') as f:
                new_metrics = json.load(f)
                new_mse = new_metrics["normalised_csd_metrics"]["mse"]

            if new_mse < best_mse:
                print(f" Improved: New MSE {new_mse:.4f} < Previous Best MSE {best_mse:.4f}")
                best_mse = new_mse

                # Save the new best results and metrics
                with open(metric_path, 'r') as f:
                    best_metrics = json.load(f)
                with open(result_path, 'r') as f:
                    best_result = json.load(f)
            else:
                print(f" No improvement (MSE: {new_mse:.4f}), retrying...")

        except Exception as e:
            print(f" Error during retry for {prefix}: {e}")
            break

        attempt += 1

    with open(metric_path, 'w') as f:
        json.dump(best_metrics, f, indent=4)
    with open(result_path, 'w') as f:
        json.dump(best_result, f, indent=4)

    print(f" Final selected MSE for Subject={subject_id}, ROI={region_index}, {exp}: {best_mse:.4f}")





if __name__ == '__main__':
    base_dir = "/Users/xuenbei/Desktop/finalyearproject/PDCM/fitted_data/"
    subject_ids = [1, 2, 3, 4, 6, 10, 11, 12, 19, 20]
    rois = list(range(100))  # 100 brain regions
    exps = ["PLCB", "LSD"]  # Experiment conditions

    # Create a list of all (subject, roi, experiment) task combinations
    tasks = [(subj, roi, exp) for subj in subject_ids for roi in rois for exp in exps]

    # Run tasks in parallel using multiprocessing
    with multiprocessing.get_context("spawn").Pool(processes=4) as pool:
        pool.starmap(partial(refit_model_if_needed, base_dir=base_dir), tasks)