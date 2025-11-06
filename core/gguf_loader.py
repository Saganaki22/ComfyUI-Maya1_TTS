"""
GGUF model loading support for Maya1 TTS.
Provides support for quantized GGUF models using llama-cpp-python.
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np


class GGUFMaya1Model:
    """
    Wrapper for Maya1 GGUF models using llama-cpp-python.
    Provides same interface as regular Maya1Model for compatibility.
    """

    def __init__(
        self,
        model,
        tokenizer,
        model_name: str,
        model_path: str,
        n_gpu_layers: int = -1
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.dtype = "gguf"
        self.device = "cuda" if n_gpu_layers > 0 else "cpu"
        self.attention_type = "gguf"

    def generate(self, input_ids: torch.Tensor, **kwargs):
        """
        Generate tokens using llama-cpp-python.

        Args:
            input_ids: Input tensor of token IDs
            **kwargs: Generation parameters

        Returns:
            Generated token IDs as tensor
        """
        # Convert input tensor to list of token IDs
        input_tokens = input_ids[0].tolist()

        # Extract generation parameters
        max_new_tokens = kwargs.get('max_new_tokens', 500)
        temperature = kwargs.get('temperature', 0.4)
        top_p = kwargs.get('top_p', 0.9)
        repetition_penalty = kwargs.get('repetition_penalty', 1.1)

        # Generate using llama-cpp
        output_tokens = []

        try:
            # Create completion
            response = self.model.create_completion(
                prompt=input_tokens,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repetition_penalty,
                stream=False
            )

            # Extract generated tokens
            if 'choices' in response and len(response['choices']) > 0:
                generated_text = response['choices'][0].get('text', '')
                # Re-tokenize to get token IDs
                output_tokens = self.tokenizer.encode(generated_text)

        except Exception as e:
            print(f"⚠️  GGUF generation error: {e}")
            raise

        # Combine input + output tokens
        all_tokens = input_tokens + output_tokens

        # Convert back to tensor
        output_tensor = torch.tensor([all_tokens], dtype=torch.long)

        return output_tensor

    def __repr__(self):
        return (f"GGUFMaya1Model(name={self.model_name}, "
                f"n_gpu_layers={self.n_gpu_layers}, "
                f"device={self.device})")


class GGUFModelLoader:
    """
    Loader for GGUF quantized Maya1 models.
    """

    _model_cache: Dict[str, GGUFMaya1Model] = {}

    @classmethod
    def load_model(
        cls,
        model_path: Path,
        n_gpu_layers: int = -1,  # -1 = all layers on GPU
        n_ctx: int = 4096,  # Context window size
        device: str = "cuda"
    ) -> GGUFMaya1Model:
        """
        Load GGUF model using llama-cpp-python.

        Args:
            model_path: Path to GGUF model file
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            n_ctx: Context window size
            device: Device preference ("cuda" or "cpu")

        Returns:
            GGUFMaya1Model wrapper
        """
        # Check cache
        cache_key = f"{model_path}|{n_gpu_layers}"
        if cache_key in cls._model_cache:
            print(f"✅ Using cached GGUF model: {model_path.name}")
            return cls._model_cache[cache_key]

        print(f"📦 Loading GGUF model: {model_path.name}")
        print(f"   GPU layers: {n_gpu_layers if n_gpu_layers >= 0 else 'all'}")
        print(f"   Context size: {n_ctx}")

        # Import llama-cpp-python
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not found. Install with:\n"
                "pip install llama-cpp-python\n"
                "For GPU support: CMAKE_ARGS=\"-DLLAMA_CUBLAS=on\" pip install llama-cpp-python --force-reinstall --no-cache-dir"
            )

        # Adjust GPU layers based on device
        if device == "cpu":
            n_gpu_layers = 0
            print("   ℹ️  CPU mode requested, setting n_gpu_layers=0")

        # Load GGUF model
        try:
            model = Llama(
                model_path=str(model_path),
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                verbose=False,
                logits_all=False  # Only need final token logits
            )
            print(f"✅ GGUF model loaded successfully")

        except Exception as e:
            print(f"❌ Failed to load GGUF model: {e}")
            raise

        # Load tokenizer from parent directory (assume same structure)
        # GGUF models should be in models/maya1-TTS/Q8.GGUF/model.gguf
        # Tokenizer should be in models/maya1-TTS/
        parent_dir = model_path.parent.parent
        tokenizer = cls._load_tokenizer(parent_dir)

        # Create wrapper
        gguf_model = GGUFMaya1Model(
            model=model,
            tokenizer=tokenizer,
            model_name=model_path.name,
            model_path=str(model_path),
            n_gpu_layers=n_gpu_layers
        )

        # Cache model
        cls._model_cache[cache_key] = gguf_model

        return gguf_model

    @staticmethod
    def _load_tokenizer(model_path: Path):
        """
        Load tokenizer from model path.

        Args:
            model_path: Path to model directory

        Returns:
            Loaded tokenizer
        """
        from transformers import AutoTokenizer

        # Check if tokenizer is in subdirectory
        if (model_path / "tokenizer").exists():
            print("   Loading tokenizer from tokenizer/ subdirectory...")
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                subfolder="tokenizer",
                trust_remote_code=True
            )
        else:
            print("   Loading tokenizer from root...")
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=True
            )

        return tokenizer

    @classmethod
    def find_gguf_models(cls, base_path: Path) -> List[Path]:
        """
        Find all GGUF models in the gguf subdirectory.

        Args:
            base_path: Base model path (e.g., models/maya1-TTS/maya1)

        Returns:
            List of GGUF model file paths
        """
        gguf_dir = base_path / "gguf"

        if not gguf_dir.exists():
            return []

        # Find all .gguf files
        gguf_files = list(gguf_dir.glob("*.gguf"))

        return sorted(gguf_files)

    @classmethod
    def clear_cache(cls):
        """Clear GGUF model cache."""
        cls._model_cache.clear()

        # Force garbage collection
        import gc
        gc.collect()
