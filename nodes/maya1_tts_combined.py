"""
Maya1 TTS Combined Node for ComfyUI.
All-in-one node with model loading and TTS generation.
"""

import torch
import numpy as np
import random
from typing import Tuple

from ..core import (
    Maya1ModelLoader,
    SNACDecoder,
    discover_maya1_models,
    get_model_path,
    get_maya1_models_dir,
    format_prompt,
    check_interruption,
    load_emotions_list
)


class Maya1TTSCombinedNode:
    """
    Combined Maya1 TTS node - loads model and generates speech in one node.

    Features:
    - Model loading with caching
    - Voice design through natural language
    - 20+ emotion tags with clickable buttons
    - Native ComfyUI cancel support
    - Real-time progress tracking
    - VRAM management
    """

    DESCRIPTION = """All-in-one Maya1 TTS: Load model and generate expressive speech.

AVAILABLE EMOTION TAGS:
<laugh>, <laugh_harder>, <giggle>, <chuckle>
<cry>, <sigh>, <gasp>, <whisper>
<angry>, <scream>, <snort>, <yawn>
<cough>, <sneeze>, <breathing>, <humming>, <throat_clearing>

USAGE EXAMPLES:
• Hello! This is Maya1 <laugh> the best voice AI!
• I'm so excited <gasp> to meet you!
• This is amazing <whisper> don't tell anyone.

VOICE DESCRIPTION:
Use natural language to describe the voice characteristics:
"Realistic male voice in the 30s with American accent. Warm timbre, conversational pacing."

GENERATION TIPS:
• Keep model in VRAM: Enable for faster repeated generations
• Temperature 0.4: Good balance between quality and variety
• Top-p 0.9: Recommended for natural speech
• Seed: Use same seed for reproducible results

MODEL SETTINGS:
• Attention mechanisms:
  - SDPA: Most compatible and fastest for TTS (default)
  - Flash Attention 2: Faster for batch inference (requires flash-attn)
  - Sage Attention: Memory efficient for long sequences (requires sageattention)

• Data types (from lowest to highest memory):
  - 4bit (BNB): NF4 quantization (~6GB VRAM, requires bitsandbytes, SLOWER than fp16/bf16)
  - 8bit (BNB): INT8 quantization (~7GB VRAM, requires bitsandbytes, SLOWER than fp16/bf16)
  - float16: 16-bit half precision (~8-9GB VRAM, FAST, good quality)
  - bfloat16: 16-bit brain float (~8-9GB VRAM, FAST, recommended, best stability)
  - float32: 32-bit full precision (~16GB VRAM, highest quality, slower)

⚠️ IMPORTANT: Quantization (4bit/8bit) is SLOWER than fp16/bf16!
   Only use quantization if you have limited VRAM (<10GB).
   If you have 10GB+ VRAM, use float16 or bfloat16 for best speed.

Note: Quantization requires CUDA and bitsandbytes: pip install bitsandbytes

Output: 24kHz mono audio ready for ComfyUI audio nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        """Define input parameters for the node."""
        return {
            "required": {
                # Model settings
                "model_name": (discover_maya1_models(), {
                    "default": discover_maya1_models()[0] if discover_maya1_models() else None
                }),
                "dtype": (["4bit (BNB)", "8bit (BNB)", "float16", "bfloat16", "float32"], {
                    "default": "bfloat16"
                }),
                "attention_mechanism": (["sdpa", "flash_attention_2", "sage_attention"], {
                    "default": "sdpa"
                }),
                "device": (["cuda", "cpu"], {
                    "default": "cuda"
                }),

                # Voice and text
                "voice_description": ("STRING", {
                    "multiline": True,
                    "default": "Realistic male voice in the 30s age with american accent. Normal pitch, warm timbre, conversational pacing.",
                    "dynamicPrompts": False
                }),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello! This is Maya1 <laugh> the best open source voice AI model with emotions.",
                    "dynamicPrompts": False
                }),

                # Generation settings
                "keep_model_in_vram": ("BOOLEAN", {
                    "default": True
                }),
                "temperature": ("FLOAT", {
                    "default": 0.4,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.05
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05
                }),
                "max_tokens": ("INT", {
                    "default": 2000,
                    "min": 100,
                    "max": 8000,
                    "step": 100
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.1,
                    "min": 1.0,
                    "max": 2.0,
                    "step": 0.05
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "audio/maya1"

    def generate_speech(
        self,
        model_name: str,
        dtype: str,
        attention_mechanism: str,
        device: str,
        voice_description: str,
        text: str,
        keep_model_in_vram: bool,
        temperature: float,
        top_p: float,
        max_tokens: int,
        repetition_penalty: float,
        seed: int,
        emotion_tag_insert: str = "(none)"
    ) -> Tuple[dict]:
        """
        Load model (if needed) and generate expressive speech.

        Returns:
            Tuple containing audio dictionary for ComfyUI
        """
        # Check for cancellation before starting
        check_interruption()

        # Simple seed logic: if seed is 0, randomize; otherwise use the provided seed
        # This way seed=0 is always random, and you can set a specific seed for reproducibility
        if seed == 0:
            actual_seed = random.randint(1, 0xffffffffffffffff)
        else:
            actual_seed = seed

        print("=" * 70)
        print("🎤 Maya1 TTS Generation")
        print("=" * 70)
        print(f"🎲 Seed: {actual_seed}")
        print(f"💾 VRAM setting: {'Keep in VRAM' if keep_model_in_vram else 'Offload after generation'}")

        # ========== MODEL LOADING ==========
        # Get the expected models directory
        models_dir = get_maya1_models_dir()

        # Validate model name
        if model_name.startswith("(No"):
            raise ValueError(
                f"No valid Maya1 models found!\n\n"
                f"Expected location: {models_dir}\n\n"
                f"Please download a model:\n"
                f"  1. Create models directory:\n"
                f"     mkdir -p {models_dir}\n\n"
                f"  2. Download Maya1:\n"
                f"     huggingface-cli download maya-research/maya1 \\\n"
                f"       --local-dir {models_dir}/maya1\n\n"
                f"  3. Restart ComfyUI to refresh the dropdown."
            )

        # Get full model path
        model_path = get_model_path(model_name)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n\n"
                f"Make sure the model is properly downloaded to:\n"
                f"  {model_path}"
            )

        # Validate that all critical files exist
        print(f"🔍 Validating model files in: {model_path}")

        critical_files = {
            "config.json": model_path / "config.json",
            "generation_config.json": model_path / "generation_config.json",
            "tokenizer_config.json": model_path / "tokenizer" / "tokenizer_config.json",
            "tokenizer.json": model_path / "tokenizer" / "tokenizer.json",
            "model weights": model_path / "model-00001-of-00002.safetensors",
        }

        missing_files = []
        for file_name, file_path in critical_files.items():
            if file_path.exists():
                print(f"   ✅ {file_name}")
            else:
                print(f"   ❌ {file_name} - MISSING!")
                missing_files.append(file_name)

        if missing_files:
            raise FileNotFoundError(
                f"Missing critical model files: {', '.join(missing_files)}\n\n"
                f"Model directory: {model_path}\n\n"
                f"Please re-download the complete model:\n"
                f"  huggingface-cli download maya-research/maya1 \\\n"
                f"    --local-dir {model_path}"
            )

        # Check device availability
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU")
            device = "cpu"

        # Strip "(BNB)" suffix from dtype labels if present
        # e.g., "4bit (BNB)" -> "4bit"
        dtype_clean = dtype.replace(" (BNB)", "")

        # Load model using the wrapper (with caching)
        # Note: Model loading details are printed by Maya1ModelLoader.load_model()
        try:
            maya1_model = Maya1ModelLoader.load_model(
                model_path=model_path,
                attention_type=attention_mechanism,
                dtype=dtype_clean,
                device=device
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Maya1 model:\n{str(e)}\n\n"
                f"Model: {model_name}\n"
                f"Attention: {attention_mechanism}\n"
                f"Dtype: {dtype_clean}\n"
                f"Device: {device}"
            )

        check_interruption()

        # ========== SPEECH GENERATION ==========
        print(f"Keep in VRAM: {keep_model_in_vram}")
        print(f"Voice: {voice_description[:60]}...")
        print(f"Text: {text[:60]}...")
        print(f"Temperature: {temperature}, Top-p: {top_p}")
        print(f"Max tokens: {max_tokens}")
        print("=" * 70)

        # Set seed for reproducibility
        torch.manual_seed(actual_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(actual_seed)

        # Format prompt using Maya1's documented format
        # DO NOT use chat template - it adds system messages that get spoken!
        print("🔤 Formatting prompt...")

        # Maya1's format from documentation: <description="..."> text_with_emotions
        prompt = f'<description="{voice_description}"> {text}'

        # Debug: Print formatted prompt
        print(f"📝 Prompt: {prompt[:150]}...")

        # Tokenize input
        inputs = maya1_model.tokenizer(prompt, return_tensors="pt")
        print(f"📊 Input token count: {inputs['input_ids'].shape[1]}")

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Check for cancellation
        check_interruption()

        # Generate with progress tracking and cancellation checks
        print(f"🎵 Generating speech (max {max_tokens} tokens)...")

        try:
            # Setup progress tracking
            from comfy.utils import ProgressBar
            progress_bar = ProgressBar(max_tokens)

            # Create stopping criteria for cancellation support
            from transformers import StoppingCriteria, StoppingCriteriaList

            class InterruptionStoppingCriteria(StoppingCriteria):
                """Custom stopping criteria that checks for ComfyUI cancellation."""
                def __init__(self, progress_bar):
                    self.progress_bar = progress_bar
                    self.current_tokens = 0
                    self.input_length = 0
                    self.start_time = None
                    self.last_print_time = None
                    self.print_interval = 0.5  # Print progress every 0.5 seconds

                def __call__(self, input_ids, scores, **kwargs):
                    import time

                    # Store input length and start time on first call
                    if self.input_length == 0:
                        self.input_length = input_ids.shape[1]
                        self.start_time = time.time()
                        self.last_print_time = self.start_time

                    # Update progress
                    new_tokens = input_ids.shape[1] - self.input_length
                    if new_tokens > self.current_tokens:
                        self.progress_bar.update(new_tokens - self.current_tokens)
                        self.current_tokens = new_tokens

                        # Print progress with it/s to console
                        current_time = time.time()
                        if current_time - self.last_print_time >= self.print_interval:
                            elapsed = current_time - self.start_time
                            it_per_sec = new_tokens / elapsed if elapsed > 0 else 0
                            print(f"   Tokens: {new_tokens}/{max_tokens} | Speed: {it_per_sec:.2f} it/s | Elapsed: {elapsed:.1f}s", end='\r')
                            self.last_print_time = current_time

                    # Check for cancellation
                    try:
                        import execution
                        if hasattr(execution, 'interruption_requested') and execution.interruption_requested():
                            print("\n🛑 Generation cancelled by user")
                            return True  # Stop generation
                    except:
                        pass

                    return False  # Continue generation

            stopping_criteria = StoppingCriteriaList([
                InterruptionStoppingCriteria(progress_bar)
            ])

            # Generate tokens with cancellation support
            # CRITICAL: Maya1 has TWO EOS tokens in generation_config.json:
            #   - 128009 (<|eot_id|>) - Text completion token
            #   - 128258 - SNAC audio completion token
            # We need to ONLY stop on 128258 (SNAC done), not 128009 (text done)
            # Otherwise the model generates text, hits 128009, and stops before SNAC codes!

            print("🎵 Generation settings:")
            print(f"   Using EOS token: 128258 (SNAC completion only)")
            print(f"   Ignoring EOS token: 128009 (text completion)")

            import time
            generation_start = time.time()

            with torch.inference_mode():
                outputs = maya1_model.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    min_new_tokens=50,  # Force at least 50 tokens to get past text phase
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=128263,  # From generation_config.json
                    eos_token_id=128258,  # ONLY stop on SNAC completion token, NOT 128009!
                    stopping_criteria=stopping_criteria,
                    use_cache=True,  # Enable KV cache for faster generation
                )

            generation_time = time.time() - generation_start

            # Check for cancellation after generation
            check_interruption()

            # Extract generated tokens (remove input tokens)
            generated_ids = outputs[0, inputs['input_ids'].shape[1]:].tolist()

            # Print final generation statistics
            final_speed = len(generated_ids) / generation_time if generation_time > 0 else 0
            print(f"\n✅ Generated {len(generated_ids)} tokens in {generation_time:.2f}s ({final_speed:.2f} it/s)")

            # Debug: Print first few generated token IDs
            print(f"🔍 First 10 generated token IDs: {generated_ids[:10]}")

            # Debug: Decode generated tokens to see what was generated
            generated_text = maya1_model.tokenizer.decode(generated_ids, skip_special_tokens=False)
            print(f"🔍 Generated text (first 100 chars): {generated_text[:100]}...")

            # Filter SNAC tokens
            from ..core.snac_decoder import filter_snac_tokens
            snac_tokens = filter_snac_tokens(generated_ids)

            if len(snac_tokens) == 0:
                raise ValueError(
                    "No SNAC audio tokens generated!\n"
                    "The model may have only generated text tokens.\n"
                    "Try adjusting the prompt or generation parameters."
                )

            print(f"🎵 Found {len(snac_tokens)} SNAC tokens ({len(snac_tokens) // 7} frames)")

            # Check for cancellation before decoding
            check_interruption()

            # Decode SNAC tokens to audio
            print("🔊 Decoding to audio...")
            audio_waveform = SNACDecoder.decode(snac_tokens, device=device)

            # Check for cancellation after decoding
            check_interruption()

            # Convert to ComfyUI audio format
            audio_tensor = torch.from_numpy(audio_waveform).float()

            # Add batch and channel dimensions: [samples] -> [1, 1, samples]
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
            elif audio_tensor.dim() == 2:
                audio_tensor = audio_tensor.unsqueeze(0)

            audio_output = {
                "waveform": audio_tensor,
                "sample_rate": 24000
            }

            print(f"✅ Generated {len(audio_waveform) / 24000:.2f}s of audio")
            print("=" * 70)

            # Handle VRAM management based on toggle
            if not keep_model_in_vram:
                print("🗑️  Offloading model from VRAM...")
                Maya1ModelLoader.clear_cache(force=True)
                print("✅ Model offloaded from VRAM")
            else:
                print("💾 Model kept in VRAM for faster next generation")

            return (audio_output,)

        except InterruptedError as e:
            # User cancelled the generation
            print(f"\n{str(e)}")
            print("=" * 70)
            # Note: VRAM cleanup handled by ComfyUI hooks
            raise

        except Exception as e:
            # Other errors
            print(f"\n❌ Generation failed: {str(e)}")
            print("=" * 70)
            # Note: VRAM cleanup handled by ComfyUI hooks
            raise


# ComfyUI node mappings
NODE_CLASS_MAPPINGS = {
    "Maya1TTS_Combined": Maya1TTSCombinedNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Maya1TTS_Combined": "Maya1 TTS (AIO)"
}
