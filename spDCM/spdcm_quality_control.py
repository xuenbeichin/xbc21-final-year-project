import os
import json
import torch
import multiprocessing
from functools import partial
from spdcm_parallel import spDCMTrainer

def refit_model_if_needed(subject_id, region_index, exp, base_dir, max_retries=3):
    """
    Refit a spDCM model for a given subject, ROI, and experimental condition if the current
    model underperforms compared to a baseline.

    Parameters:
        subject_id (int): Subject identifier
        region_index (int): ROI index
        exp (str): Experimental condition (e.g., "PLCB", "LSD")
        base_dir (str): Path to the directory containing model outputs
        max_retries (int): Number of attempts to re-optimize the model using Optuna
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Construct paths
    sub_folder = f"sub-{subject_id:03d}"
    prefix = f"roi-{region_index}-{exp}"
    root = os.path.join(base_dir, sub_folder)

    result_path = os.path.join(root, f"{prefix}_results.json")
    metric_path = os.path.join(root, f"{prefix}_metrics.json")
    baseline_metrics_path = os.path.join(root, f"{prefix}_baseline_metrics.json")
    baseline_results_path = os.path.join(root, f"{prefix}_baseline_results.json")

    # Check for required files
    if not (os.path.exists(result_path) and os.path.exists(metric_path) and
            os.path.exists(baseline_metrics_path) and os.path.exists(baseline_results_path)):
        print(f" Missing files for Subject {subject_id}, ROI {region_index}, {exp}")
        return

    # Load baseline metrics and parameters
    try:
        with open(baseline_metrics_path, 'r') as f:
            baseline_metrics = json.load(f)
            baseline_mse = baseline_metrics["normalised_csd_metrics"]["mse"]

        with open(baseline_results_path, 'r') as f:
            baseline_results = json.load(f)
            baseline_params = baseline_results["final_parameters"]
    except Exception as e:
        print(f" Could not load baseline for {prefix}: {e}")
        return

    # Load current best metrics if they exist
    try:
        with open(metric_path, 'r') as f:
            best_metrics = json.load(f)
            best_mse = best_metrics["normalised_csd_metrics"]["mse"]
            best_nmse = best_metrics["normalised_csd_metrics"]["nmse"]
    except:
        # Use infinity as a fallback if no previous metrics
        best_mse = float("inf")
        best_nmse = float("inf")
        best_metrics = {}

    # Skip refitting if the current model is already better than baseline
    #  if best_mse < baseline_mse and best_nmse <= 1:
    if best_mse < baseline_mse and best_mse < 50 and best_nmse <= 1:
        print(f" Already good: Subject={subject_id}, ROI={region_index}, Exp={exp}")
        return

    best_result = None
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            best_result = json.load(f)

    # Attempt to re-optimize the model with Optuna
    success = False
    for attempt in range(max_retries):
        try:
            print(f" Attempt {attempt + 1}: Subject={subject_id}, ROI={region_index}, Exp={exp}")
            trainer = spDCMTrainer(subject_id, region_index, exp, device)
            best_params = trainer.run_optuna(n_trials=200)  # Search best hyperparameters
            trainer.build_model(best_params)
            trainer.train_model(num_epochs=1500)
            trainer.save_outputs()

            # Load new metrics and compare
            with open(metric_path, 'r') as f:
                new_metrics = json.load(f)
            new_mse = new_metrics["normalised_csd_metrics"]["mse"]
            new_nmse = new_metrics["normalised_csd_metrics"]["nmse"]

            if new_mse < best_mse:
                best_mse = new_mse
                best_nmse = new_nmse
                best_metrics = new_metrics
                with open(result_path, 'r') as f:
                    best_result = json.load(f)
                print("Improved, updated best model")
                success = True
                break
            else:
                print("No improvement")

        except Exception as e:
            print(f" Failed on attempt {attempt + 1}: {e}")

    # Fallback: retrain with baseline parameters using AdamW optimizer
    if not success:
        try:
            print(f"Falling back to AdamW on baseline for {prefix}")
            trainer = spDCMTrainer(subject_id, region_index, exp, device)
            trainer.build_model(baseline_params)
            trainer.train_model(num_epochs=1500)
            trainer.save_outputs()

            with open(metric_path, 'r') as f:
                best_metrics = json.load(f)
            with open(result_path, 'r') as f:
                best_result = json.load(f)

            print("Fallback finished")

        except Exception as e:
            print(f"Fallback training failed for {prefix}: {e}")
            return

    # Save updated best results
    if best_result and best_metrics:
        with open(result_path, 'w') as f:
            json.dump(best_result, f, indent=4)
        with open(metric_path, 'w') as f:
            json.dump(best_metrics, f, indent=4)

    print(f"Final result: Subject={subject_id}, ROI={region_index}, Exp={exp} | Best MSE={best_mse:.4f}, NMSE={best_nmse:.4f}")


def main():
    base_dir = "/Users/xuenbei/Desktop/finalyearproject/spDCM/fitted_data"
    subject_ids = [1, 2, 3, 4, 6, 9, 10, 11, 12, 13, 15, 17, 18, 19, 20]  # List of subjects
    rois = list(range(100))  # List of ROI indices
    exps = ["PLCB", "LSD"]  # Experimental conditions

    # Create all tasks (subject, ROI, exp combinations)
    tasks = [(subj, roi, exp) for subj in subject_ids for roi in rois for exp in exps]

    # Use multiprocessing to parallelize processing
    with multiprocessing.get_context("spawn").Pool(processes=4) as pool:
        pool.starmap(partial(refit_model_if_needed, base_dir=base_dir), tasks)


if __name__ == '__main__':
    main()
