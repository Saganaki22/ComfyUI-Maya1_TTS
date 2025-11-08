"""
Core modules for Maya1 TTS ComfyUI integration.
"""

from .model_wrapper import Maya1Model, Maya1ModelLoader
from .snac_decoder import SNACDecoder
from .chunking import (
    smart_chunk_text,
    estimate_tokens_for_text,
    should_chunk_text
)
from .utils import (
    discover_maya1_models,
    get_model_path,
    get_maya1_models_dir,
    load_emotions_list,
    format_prompt,
    check_interruption,
    ProgressCallback,
    crossfade_audio
)

__all__ = [
    "Maya1Model",
    "Maya1ModelLoader",
    "SNACDecoder",
    "smart_chunk_text",
    "estimate_tokens_for_text",
    "should_chunk_text",
    "discover_maya1_models",
    "get_model_path",
    "get_maya1_models_dir",
    "load_emotions_list",
    "format_prompt",
    "check_interruption",
    "ProgressCallback",
    "crossfade_audio",
]
