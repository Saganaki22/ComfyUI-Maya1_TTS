"""
Utility functions for Maya1 TTS ComfyUI nodes.
Includes ComfyUI-native cancel support and progress tracking.
"""

import os
from pathlib import Path
from typing import List, Optional


def get_maya1_models_dir() -> Path:
    """
    Get the Maya1 models directory within ComfyUI's models folder.

    Returns:
        Path to ComfyUI/models/maya1-TTS/
    """
    try:
        # Try to use ComfyUI's folder_paths
        import folder_paths
        comfyui_models_dir = Path(folder_paths.models_dir)
    except:
        # Fallback: try to detect ComfyUI directory
        # Look for ComfyUI installation in common locations
        current_file = Path(__file__).resolve()

        # Navigate up from custom_nodes/ComfyUI-Maya1_TTS/core/utils.py
        # to find ComfyUI root (should have a 'models' folder)
        for parent in current_file.parents:
            if (parent / "models").exists() and (parent / "custom_nodes").exists():
                comfyui_models_dir = parent / "models"
                break
        else:
            # Ultimate fallback: use current directory
            comfyui_models_dir = Path.cwd() / "models"

    maya1_models_dir = comfyui_models_dir / "maya1-TTS"
    return maya1_models_dir


def discover_maya1_models() -> List[str]:
    """
    Scan ComfyUI/models/maya1-TTS/ for available Maya1 models.

    Returns:
        List of model directory names (relative to ComfyUI/models/maya1-TTS/)
    """
    models_dir = get_maya1_models_dir()

    if not models_dir.exists():
        print(f"⚠️  Maya1 models directory not found: {models_dir}")
        print(f"💡 Create it and download models with:")
        print(f"   mkdir -p {models_dir}")
        print(f"   huggingface-cli download maya-research/maya1 --local-dir {models_dir}/maya1")
        return ["(No models folder found - see console for instructions)"]

    # Find directories with config.json (HuggingFace model format)
    models = []
    for item in models_dir.iterdir():
        if item.is_dir():
            # Check for config.json in root or in a checkpoint subdirectory
            if (item / "config.json").exists():
                models.append(item.name)
            elif any((item / d / "config.json").exists() for d in ["checkpoint-*"] if (item / d).exists()):
                models.append(item.name)

    if not models:
        print(f"⚠️  No valid Maya1 models found in {models_dir}")
        print(f"💡 Download a model with:")
        print(f"   huggingface-cli download maya-research/maya1 --local-dir {models_dir}/maya1")
        return ["(No valid models found - see console for instructions)"]

    return sorted(models)


def get_model_path(model_name: str) -> Path:
    """
    Get the full path to a model directory.

    Args:
        model_name: Name of the model folder

    Returns:
        Full path to the model directory
    """
    return get_maya1_models_dir() / model_name


def load_emotions_list() -> List[str]:
    """
    Load the list of supported emotion tags from resources/emotions.txt.

    Returns:
        List of emotion tag names (without angle brackets)
    """
    emotions_file = Path(__file__).parent.parent / "resources" / "emotions.txt"

    if not emotions_file.exists():
        # Fallback list if file doesn't exist
        return [
            "laugh", "laugh_harder", "giggle", "chuckle", "cry", "sigh",
            "gasp", "whisper", "angry", "scream", "snort", "yawn",
            "cough", "sneeze", "breathing", "humming", "throat_clearing"
        ]

    with open(emotions_file, 'r') as f:
        emotions = [line.strip() for line in f if line.strip()]

    return emotions


def format_prompt(voice_description: str, text: str) -> str:
    """
    Format the prompt using Maya1's expected format with chat template.

    Args:
        voice_description: Natural language voice description
        text: Text to synthesize (may contain emotion tags)

    Returns:
        Formatted prompt string
    """
    # Maya1 uses a chat-like format with system/user messages
    # The voice description acts as the "system" instruction
    # The text to synthesize is the "user" message

    # Format as a conversation to trigger audio generation
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a voice synthesis system. Generate natural speech audio using SNAC codes for the following voice characteristics: {voice_description}<|eot_id|><|start_header_id|>user<|end_header_id|>

{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    return prompt


def check_interruption():
    """
    Check if ComfyUI has requested interruption.
    Raises an exception if cancellation was requested.

    This integrates with ComfyUI's native cancel functionality.
    """
    try:
        # Try to import ComfyUI's execution module
        import execution
        if hasattr(execution, 'interruption_requested') and execution.interruption_requested():
            raise InterruptedError("🛑 Generation cancelled by user")
    except ImportError:
        # If ComfyUI modules aren't available (e.g., testing), just continue
        pass
    except InterruptedError:
        # Re-raise interruption errors
        raise
    except Exception as e:
        # Silently ignore other errors (module might not have the attribute in older versions)
        pass


class ProgressCallback:
    """
    Progress tracking callback for ComfyUI integration.
    Shows generation progress in the ComfyUI UI.
    """

    def __init__(self, total_steps: int, desc: str = "Generating"):
        self.total_steps = total_steps
        self.current_step = 0
        self.desc = desc
        self.pbar = None

        # Try to use ComfyUI's progress bar
        try:
            from comfy.utils import ProgressBar
            self.pbar = ProgressBar(total_steps)
        except ImportError:
            # Fallback: just print progress
            self.pbar = None

    def update(self, steps: int = 1):
        """Update progress by the specified number of steps."""
        self.current_step += steps

        if self.pbar is not None:
            self.pbar.update(steps)
        else:
            # Fallback: print percentage
            if self.current_step % max(1, self.total_steps // 10) == 0:
                pct = (self.current_step / self.total_steps) * 100
                print(f"⏳ {self.desc}: {pct:.1f}%")

        # Check for cancellation on each update
        check_interruption()

    def close(self):
        """Close the progress bar."""
        if self.pbar is not None:
            self.pbar.update(self.total_steps - self.current_step)


def crossfade_audio(audio1, audio2, crossfade_samples: int = 1200):
    """
    Crossfade two audio arrays for smooth transitions.

    Args:
        audio1: First audio array (numpy or torch)
        audio2: Second audio array (numpy or torch)
        crossfade_samples: Number of samples to crossfade (default 1200 = 50ms at 24kHz)

    Returns:
        Crossfaded audio array
    """
    import numpy as np
    import torch

    # Convert to numpy for processing
    is_torch = False
    if isinstance(audio1, torch.Tensor):
        is_torch = True
        audio1_np = audio1.cpu().numpy()
        audio2_np = audio2.cpu().numpy()
    else:
        audio1_np = audio1
        audio2_np = audio2

    # Handle different shapes: [batch, channels, samples] or [samples]
    if audio1_np.ndim == 3:
        # Shape: [batch, channels, samples]
        batch, channels, samples1 = audio1_np.shape
        samples2 = audio2_np.shape[2]

        # Ensure crossfade_samples doesn't exceed audio length
        crossfade_samples = min(crossfade_samples, samples1, samples2)

        if crossfade_samples > 0:
            # Create fade curves
            fade_out = np.linspace(1.0, 0.0, crossfade_samples).reshape(1, 1, -1)
            fade_in = np.linspace(0.0, 1.0, crossfade_samples).reshape(1, 1, -1)

            # Apply crossfade to overlapping region
            audio1_fade = audio1_np.copy()
            audio1_fade[:, :, -crossfade_samples:] *= fade_out

            audio2_fade = audio2_np.copy()
            audio2_fade[:, :, :crossfade_samples] *= fade_in

            # Combine: audio1 (minus fade region) + crossfade + audio2 (minus fade region)
            result = np.concatenate([
                audio1_fade[:, :, :-crossfade_samples],
                audio1_fade[:, :, -crossfade_samples:] + audio2_fade[:, :, :crossfade_samples],
                audio2_fade[:, :, crossfade_samples:]
            ], axis=2)
        else:
            # No crossfade, just concatenate
            result = np.concatenate([audio1_np, audio2_np], axis=2)

    elif audio1_np.ndim == 1:
        # Shape: [samples]
        samples1 = len(audio1_np)
        samples2 = len(audio2_np)

        crossfade_samples = min(crossfade_samples, samples1, samples2)

        if crossfade_samples > 0:
            fade_out = np.linspace(1.0, 0.0, crossfade_samples)
            fade_in = np.linspace(0.0, 1.0, crossfade_samples)

            audio1_fade = audio1_np.copy()
            audio1_fade[-crossfade_samples:] *= fade_out

            audio2_fade = audio2_np.copy()
            audio2_fade[:crossfade_samples] *= fade_in

            result = np.concatenate([
                audio1_fade[:-crossfade_samples],
                audio1_fade[-crossfade_samples:] + audio2_fade[:crossfade_samples],
                audio2_fade[crossfade_samples:]
            ])
        else:
            result = np.concatenate([audio1_np, audio2_np])

    else:
        raise ValueError(f"Unexpected audio shape: {audio1_np.shape}")

    # Convert back to torch if needed
    if is_torch:
        result = torch.from_numpy(result)

    return result
