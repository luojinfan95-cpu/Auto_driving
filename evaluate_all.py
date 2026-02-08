import os
import glob
import argparse
import numpy as np
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate all models excluding error frames")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory with model result folders")
    return parser.parse_args()

def collect_errors(scene_files, model_name):
    error_set = set()
    error_entries = []
    for scene_file in scene_files:
        scene_data = load_json_file(scene_file)
        scene_name = scene_data["scene_info"]["name"]
        for frame in scene_data["frames"]:
            frame_index = frame["frame_index"]
            pred_actions_str = frame["inference"]["pred_actions_str"]
            pred_actions = extract_driving_action(pred_actions_str, error_handling=True)
            if not pred_actions:
                error_set.add((scene_name, frame_index))
                error_entries.append((model_name, scene_name, frame_index, pred_actions_str))
    return error_set, error_entries

def evaluate(scene_files, error_set):
    """Evaluate metrics for a given set of scene files, skipping error frames."""
    ade_1s, ade_2s, ade_3s, ade_avg, fde, miss_rate = [], [], [], [], [], []

    total_token_usage = {
        "scene_prompt": {"input": 0, "output": 0},
        "intent_prompt": {"input": 0, "output": 0},
        "waypoint_prompt": {"input": 0, "output": 0},
        "total": {"input": 0, "output": 0},
    }
    total_time_usage = {
        "scene_prompt": 0,
        "intent_prompt": 0,
        "waypoint_prompt": 0,
        "total": 0,
    }

    successful_frame_count = 0

    for scene_file in scene_files:
        scene_data = load_json_file(scene_file)
        scene_name = scene_data["scene_info"]["name"]
        for frame in scene_data["frames"]:
            frame_index = frame["frame_index"]
            if (scene_name, frame_index) in error_set:
                continue

            gt_positions = frame["ego_info"]["gt_positions"]
            pred_actions_str = frame["inference"]["pred_actions_str"]
            pred_actions = extract_driving_action(pred_actions_str, error_handling=True)

            if not pred_actions:
                continue

            pred_trajectory = integrate_driving_commands(pred_actions, dt=0.5)
            metrics = compute_metrics(pred_trajectory, gt_positions)

            if metrics["ADE_1s"] is not None: ade_1s.append(metrics["ADE_1s"])
            if metrics["ADE_2s"] is not None: ade_2s.append(metrics["ADE_2s"])
            if metrics["ADE_3s"] is not None: ade_3s.append(metrics["ADE_3s"])
            if metrics["ADE_avg"] is not None: ade_avg.append(metrics["ADE_avg"])
            if metrics["FDE"] is not None: fde.append(metrics["FDE"])
            if metrics["missRate_2"] is not None: miss_rate.append(metrics["missRate_2"])

            successful_frame_count += 1

            for k, v in frame["token_usage"].items():
                total_token_usage[k]["input"] += v["input"]
                total_token_usage[k]["output"] += v["output"]
                total_token_usage["total"]["input"] += v["input"]
                total_token_usage["total"]["output"] += v["output"]

            for k, v in frame["time_usage"].items():
                total_time_usage[k] += v
                total_time_usage["total"] += v

    avg_token_usage = {
        k: {
            "input": v["input"] / successful_frame_count if successful_frame_count > 0 else 0,
            "output": v["output"] / successful_frame_count if successful_frame_count > 0 else 0,
        }
        for k, v in total_token_usage.items()
    }

    avg_time_usage = {
        k: v / successful_frame_count if successful_frame_count > 0 else 0
        for k, v in total_time_usage.items()
    }

    metrics = {
        "ADE_1s": np.mean(ade_1s).item() if ade_1s else None,
        "ADE_2s": np.mean(ade_2s).item() if ade_2s else None,
        "ADE_3s": np.mean(ade_3s).item() if ade_3s else None,
        "ADE_avg": np.mean(ade_avg).item() if ade_avg else None,
        "FDE": np.mean(fde).item() if fde else None,
        "missRate_2": np.mean(miss_rate).item() if miss_rate else None,
    }

    return metrics, avg_token_usage, avg_time_usage

def main():
    args = parse_args()

    model_dirs = [
        os.path.join(args.results_dir, d)
        for d in os.listdir(args.results_dir)
        if os.path.isdir(os.path.join(args.results_dir, d))
        and os.path.exists(os.path.join(args.results_dir, d, "output"))
    ]

    all_errors = set()
    all_error_entries = []
    all_frames = set()  # store (scene, frame) pairs globally

    # Pass 1: Collect all errors and track all frames
    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        scene_files = glob.glob(os.path.join(model_dir, "output", "*.json"))
        print(f"Scanning {model_name} for parse errors...")
        errors, entries = collect_errors(scene_files, model_name)
        all_errors.update(errors)
        all_error_entries.extend(entries)

        for scene_file in scene_files:
            scene_data = load_json_file(scene_file)
            scene_name = scene_data["scene_info"]["name"]
            for frame in scene_data["frames"]:
                frame_index = frame["frame_index"]
                all_frames.add((scene_name, frame_index))

    print(f"\nCollected {len(all_errors)} unique error (scene, frame) pairs.\n")

    # Write shared error_all.txt
    if all_error_entries:
        error_path = os.path.join(args.results_dir, "error_all.txt")
        with open(error_path, "w") as f:
            f.write(f"{'model':<25}{'scene_name':<15}{'frame_index':<15}{'pred_actions_str'}\n")
            for model, scene, idx, pred_str in all_error_entries:
                f.write(f"{model:<25}{scene:<15}{idx:<15}{pred_str}\n")
        print(f"Saved all error entries to {error_path}")

    # Final global counts (consistent across all models)
    frames_original = len(all_frames)
    frames_successful = len(all_frames - all_errors)

    # Pass 2: Evaluate each model separately (excluding error frames)
    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        print(f"Evaluating {model_name} (excluding error frames)...")
        scene_files = glob.glob(os.path.join(model_dir, "output", "*.json"))
        metrics, token_usage, time_usage = evaluate(scene_files, all_errors)

        final_output = {
            "frames_original": frames_original,
            "frames_successful": frames_successful,
            "metrics": metrics,
            "token_usage": token_usage,
            "time_usage": time_usage,
        }

        analysis_dir = os.path.join(model_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        output_path = os.path.join(analysis_dir, "evaluate_all.json")
        save_dict_to_json(final_output, output_path)
        print(f"Saved results to {output_path}\n")

    print(f"\nFinal Evaluation Summary (shared across all models):")
    print(f"Frames original: {frames_original}")
    print(f"Frames successful: {frames_successful}")

if __name__ == "__main__":
    main()
