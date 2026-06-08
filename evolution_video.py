import os
import re
import glob
import yaml
import argparse
from pathlib import Path
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.VideoClip import ColorClip, TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip, clips_array, concatenate_videoclips
from gymnasium.wrappers import RecordVideo

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from environment import make_highway_env
from agent import load_agent

CHECKPOINT_PATTERN = "highway_ppo_model_*_steps.zip"
FINAL_MODEL_NAME = "highway_ppo_final.zip"


class RandomAgent:
    """Simple random policy used to produce an untrained baseline."""

    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, obs, deterministic=True):
        return self.action_space.sample(), None


def parse_checkpoint_steps(path: Path):
    match = re.search(r"_(\d+)_steps\.zip$", path.name)
    return int(match.group(1)) if match else None


def find_stage_models(checkpoint_dir: Path):
    checkpoint_dir = checkpoint_dir.resolve()
    model_paths = sorted(checkpoint_dir.glob(CHECKPOINT_PATTERN), key=os.path.getmtime)
    step_models = [(parse_checkpoint_steps(p), p) for p in model_paths if parse_checkpoint_steps(p) is not None]
    step_models.sort(key=lambda x: x[0])

    final_model = checkpoint_dir / FINAL_MODEL_NAME
    if not final_model.exists():
        final_model = None

    half_model = None
    if len(step_models) >= 2:
        # Choose the earlier checkpoint for the half-trained stage when only a few checkpoints exist.
        half_index = (len(step_models) - 1) // 2
        half_model = step_models[half_index][1]
    elif len(step_models) == 1:
        half_model = step_models[0][1]

    return {
        "half": half_model,
        "final": final_model,
        "all_step_models": [p for _, p in step_models]
    }


def make_stage_video(stage_name, model_path, config_path, output_dir, episodes):
    stage_dir = output_dir / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)

    for old_video in stage_dir.glob("*.mp4"):
        old_video.unlink()

    env = make_highway_env(config_path, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=str(stage_dir),
        episode_trigger=lambda _: True,
        name_prefix=stage_name
    )

    if model_path is None:
        print("Creating random baseline for stage:", stage_name)
        model = RandomAgent(env.action_space)
    else:
        print(f"Loading stage {stage_name} model from {model_path}")
        model = load_agent(str(model_path), env=env)

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            result = env.step(action)
            if len(result) == 5:
                obs, reward, done, truncated, info = result
            else:
                obs, reward, done, info = result

        print(f"Completed {stage_name} episode {ep + 1}")

    env.close()

    recorded_files = sorted(stage_dir.glob(f"{stage_name}*.mp4"), key=os.path.getmtime)
    if not recorded_files:
        raise FileNotFoundError(f"No video produced for stage {stage_name}")

    return recorded_files[-1]


def make_labeled_clip(clip, label):
    try:
        label_text = TextClip(label, fontsize=30, color="white", bg_color="black", size=(clip.w, 50))
        label_text = label_text.set_duration(clip.duration).set_fps(clip.fps)
        return CompositeVideoClip([clip, label_text.set_position(("center", "top"))])
    except Exception:
        return clip


def build_evolution_video(stage_paths, labels, output_path):
    stage_clips = []
    for path, label in zip(stage_paths, labels):
        clip = VideoFileClip(str(path))
        labeled = make_labeled_clip(clip, label)
        stage_clips.append(labeled)

    min_duration = min(clip.duration for clip in stage_clips)
    # Trim all stage clips to the same duration using `subclip`
    stage_clips = [clip.subclipped(0, min_duration) for clip in stage_clips]

    min_height = min(clip.h for clip in stage_clips)
    stage_clips = [clip.resized(height=min_height) for clip in stage_clips]

    combined = clips_array([stage_clips])
    # Preserve original FPS to avoid speed distortion
    combined.write_videofile(str(output_path), fps=stage_clips[0].fps, codec="libx264", audio=False)
    combined.close()
    for clip in stage_clips:
        clip.close()


def main():
    parser = argparse.ArgumentParser(description="Generate a side-by-side evolution video from untrained, half-trained, and fully trained agents.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Path to checkpoint directory")
    parser.add_argument("--output_dir", type=str, default="videos/eval", help="Path to save videos")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes per stage")
    args = parser.parse_args()

    config_path = Path(args.config)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    total_timesteps = int(config.get("training", {}).get("total_timesteps", 500000))

    stage_models = find_stage_models(checkpoint_dir)
    stages = [("untrained", None)]

    if stage_models["half"] is not None:
        stages.append(("half_trained", stage_models["half"]))
    else:
        print("Warning: no intermediate checkpoint found for half-trained stage.")

    if stage_models["final"] is not None:
        stages.append((f"fully_trained_{total_timesteps}_steps", stage_models["final"]))
    elif stage_models["all_step_models"]:
        stages.append(("fully_trained", stage_models["all_step_models"][-1]))
    else:
        raise FileNotFoundError("No fully trained model available.")

    stage_video_paths = []
    stage_video_labels = []
    for stage_name, model_path in stages:
        stage_video = make_stage_video(stage_name, model_path, config_path, output_dir, args.episodes)
        stage_video_paths.append(stage_video)
        
        # Create simple label for this stage
        if stage_name == "untrained":
            label = "Untrained"
        elif stage_name == "half_trained":
            label = "Half Trained"
        else:
            label = "Trained"
        stage_video_labels.append(label)

    evolution_path = output_dir / "evolution.mp4"
    print(f"Building side-by-side evolution video at {evolution_path}")
    build_evolution_video(stage_video_paths, stage_video_labels, evolution_path)
    print("Done. Evolution video saved to:", evolution_path)


if __name__ == "__main__":
    main()
