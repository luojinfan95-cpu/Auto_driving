import os
import numpy as np
import argparse
import glob
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="LightEMMA: Evaluation")
    parser.add_argument("--results_dir", type=str, default='results/gpt-4o')
    parser.add_argument("--error_handling", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def evaluate(args, scene_files, viz_dir, data_dir):
    total_frames = 0
    successful_frames = 0
    parse_error_frames = 0

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

    error_entries = []

    for scene_file in scene_files:
        scene_data = load_json_file(scene_file)
        scene_name = scene_data["scene_info"]["name"]

        for frame in scene_data["frames"]:
            total_frames += 1
            frame_index = frame["frame_index"]

            gt_positions = frame["ego_info"]["gt_positions"]
            pred_actions_str = frame["inference"]["pred_actions_str"]
            pred_actions = extract_driving_action(pred_actions_str, args.error_handling)

            if not pred_actions:
                parse_error_frames += 1
                error_entries.append((scene_name, frame_index, pred_actions_str))
                continue

            pred_trajectory = integrate_driving_commands(pred_actions, dt=0.5)
            metrics = compute_metrics(pred_trajectory, gt_positions)
            successful_frames += 1

            if metrics["ADE_1s"] is not None: ade_1s.append(metrics["ADE_1s"])
            if metrics["ADE_2s"] is not None: ade_2s.append(metrics["ADE_2s"])
            if metrics["ADE_3s"] is not None: ade_3s.append(metrics["ADE_3s"])
            if metrics["ADE_avg"] is not None: ade_avg.append(metrics["ADE_avg"])
            if metrics["FDE"] is not None: fde.append(metrics["FDE"])
            if metrics["missRate_2"] is not None: miss_rate.append(metrics["missRate_2"])

            for k, v in frame["token_usage"].items():
                total_token_usage[k]["input"] += v["input"]
                total_token_usage[k]["output"] += v["output"]
                total_token_usage["total"]["input"] += v["input"]
                total_token_usage["total"]["output"] += v["output"]

            for k, v in frame["time_usage"].items():
                total_time_usage[k] += v
                total_time_usage["total"] += v

            if args.visualize:
                image_path = os.path.join(data_dir, "samples/CAM_FRONT", frame["image_name"])
                if os.path.exists(image_path):
                    camera_params = frame["camera_params"]
                    viz_filename = f"{scene_name}_frame{frame_index}.png"
                    viz_path = os.path.join(viz_dir, viz_filename)
                    OverlayTrajectory(
                        img_path=image_path,
                        wp_world1=gt_positions,
                        wp_world2=pred_trajectory,
                        cam_to_ego=camera_params,
                        ego_pos=(0, 0),
                        ego_heading=0.0,
                        save_path=viz_path,
                    )
                else:
                    print(f"{image_path} not found, continue...")

    avg_token_usage = {
        k: {
            "input": v["input"] / successful_frames if successful_frames > 0 else 0,
            "output": v["output"] / successful_frames if successful_frames > 0 else 0,
        }
        for k, v in total_token_usage.items()
    }

    avg_time_usage = {
        k: v / successful_frames if successful_frames > 0 else 0
        for k, v in total_time_usage.items()
    }

    metrics = {
        "frames_total": total_frames,
        "frames_successful": successful_frames,
        "frames_parse_errors": parse_error_frames,
        "success_rate": successful_frames / total_frames if total_frames > 0 else 0,
        "metrics": {
            "ADE_1s": np.mean(ade_1s).item() if ade_1s else None,
            "ADE_2s": np.mean(ade_2s).item() if ade_2s else None,
            "ADE_3s": np.mean(ade_3s).item() if ade_3s else None,
            "ADE_avg": np.mean(ade_avg).item() if ade_avg else None,
            "FDE": np.mean(fde).item() if fde else None,
            "missRate_2": np.mean(miss_rate).item() if miss_rate else None,
        },
        "token_usage": avg_token_usage,
        "time_usage": avg_time_usage,
    }

    return metrics, error_entries


def main():
    args = parse_args()
    config = load_config("config.yaml")
    data_dir = config["data"]["root"]
    results_dir = args.results_dir

    output_dir = os.path.join(results_dir, "output")
    scene_files = glob.glob(os.path.join(output_dir, "*.json"))
    if not scene_files:
        print(f"No JSON files found in {output_dir}")
        return

    viz_dir = os.path.join(results_dir, "visualize")
    os.makedirs(viz_dir, exist_ok=True)

    metrics, error_entries = evaluate(args, scene_files, viz_dir, data_dir)

    print("\nOverall Metrics: \n")
    print(f"ADE 1s: {metrics['metrics']['ADE_1s']:.4f}")
    print(f"ADE 2s: {metrics['metrics']['ADE_2s']:.4f}")
    print(f"ADE 3s: {metrics['metrics']['ADE_3s']:.4f}")
    print(f"ADE avg: {metrics['metrics']['ADE_avg']:.4f}")
    print(f"FDE: {metrics['metrics']['FDE']:.4f}")
    print(f"missRate_2: {metrics['metrics']['missRate_2']:.4f}")

    success = metrics["frames_successful"]
    token_usage = metrics["token_usage"]["total"]

    print(f"\nInput tokens per frame: {token_usage['input']:.2f}")
    print(f"Output tokens per frame: {token_usage['output']:.2f}")

    time_usage = metrics["time_usage"]["total"]
    print(f"Inference time per frame: {time_usage:.2f} seconds")

    print(
        f"\nFrames with errors: {metrics['frames_parse_errors']}/{metrics['frames_total']} "
        f"({metrics['frames_parse_errors'] / metrics['frames_total'] * 100:.2f}%)"
    )

    analysis_dir = os.path.join(results_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    save_dict_to_json({"metrics": metrics}, os.path.join(analysis_dir, "evaluation.json"))
    print(f"\nSaved overall evaluation to {os.path.join(analysis_dir, 'evaluation.json')}")

    if error_entries:
        error_log_path = os.path.join(analysis_dir, "error.txt")
        with open(error_log_path, "w") as f:
            f.write(f"{'scene_name':<15}{'frame_index':<15}{'pred_actions_str'}\n")
            for scene_name, frame_index, pred_str in error_entries:
                f.write(f"{scene_name:<20}{frame_index:<10}{pred_str}\n")
        print(f"Saved error cases to {error_log_path}")


if __name__ == "__main__":
    main()
