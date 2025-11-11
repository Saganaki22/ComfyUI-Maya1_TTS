"""
Maya1 TTS Combined Node for ComfyUI.
All-in-one node with model loading and TTS generation.
"""

import torch
import numpy as np
import random
import re
import gc
from typing import Tuple, List
import comfy.model_management as mm

from ..core import (
    Maya1ModelLoader,
    SNACDecoder,
    discover_maya1_models,
    get_model_path,
    get_maya1_models_dir,
    format_prompt,
    load_emotions_list,
    crossfade_audio
)


def create_progress_bar(current: int, total: int, width: int = 12, show_numbers: bool = True) -> str:
    """
    Create a visual progress bar like ComfyUI's native one.

    Args:
        current: Current progress value
        total: Total value
        width: Width of the progress bar in characters
        show_numbers: Whether to show the numbers after the bar

    Returns:
        Formatted progress bar string
    """
    if total == 0:
        percent = 0
    else:
        percent = min(current / total, 1.0)

    filled = int(width * percent)
    empty = width - filled

    bar = '█' * filled + '░' * empty

    if show_numbers:
        return f"[{bar}] {current}/{total}"
    else:
        return f"[{bar}]"

def split_text_smartly(text: str, max_words_per_chunk: int = 100) -> List[str]:
    """
    Split text into chunks at sentence boundaries, keeping emotion tags intact.
    Improved to NEVER cut words mid-sentence.

    Args:
        text: Input text to split
        max_words_per_chunk: Maximum words per chunk (default 100)

    Returns:
        List of text chunks
    """
    # Better sentence boundary detection that handles emotion tags
    # Split on: . ! ? followed by whitespace (and optionally capital letter or end of string)
    # This regex keeps the punctuation with the sentence
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z<]|$)'
    sentences = re.split(sentence_pattern, text.strip())

    # Clean up empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    # Group sentences into chunks
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        # Count words (emotion tags don't count as words)
        # Remove emotion tags temporarily for word count
        text_without_tags = re.sub(r'<[^>]+>', '', sentence)
        word_count = len(text_without_tags.split())

        # If single sentence exceeds max, split on commas or semicolons
        if word_count > max_words_per_chunk:
            # Split long sentence on commas, keeping punctuation
            parts = re.split(r'(,\s+|;\s+)', sentence)

            for i, part in enumerate(parts):
                if not part.strip():
                    continue

                # For delimiters (commas/semicolons), add to previous chunk
                if part.strip() in [',', ';']:
                    if current_chunk:
                        current_chunk[-1] += part
                    continue

                # Count words in this part
                part_text = re.sub(r'<[^>]+>', '', part)
                part_words = len(part_text.split())

                if current_word_count + part_words > max_words_per_chunk and current_chunk:
                    # Start new chunk
                    chunks.append(''.join(current_chunk))
                    current_chunk = [part]
                    current_word_count = part_words
                else:
                    # Add to current chunk
                    if current_chunk and not current_chunk[-1].endswith((' ', ',', ';')):
                        current_chunk.append(' ')
                    current_chunk.append(part)
                    current_word_count += part_words
        else:
            # Normal sentence handling
            if current_word_count + word_count > max_words_per_chunk and current_chunk:
                # Save current chunk and start new one
                chunks.append(''.join(current_chunk))
                current_chunk = [sentence]
                current_word_count = word_count
            else:
                # Add to current chunk with space
                if current_chunk:
                    current_chunk.append(' ')
                current_chunk.append(sentence)
                current_word_count += word_count

    # Add remaining chunk
    if current_chunk:
        chunks.append(''.join(current_chunk))

    return chunks if chunks else [text]


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

    DESCRIPTION = ""

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
                "attention_mechanism": (["sdpa", "eager", "flash_attention_2", "sage_attention"], {
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
                    "default": 0.4,  # Official Maya1 recommendation (from transformers_inference.py)
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
                "max_new_tokens": ("INT", {
                    "default": 4000,
                    "min": 100,
                    "max": 16000,
                    "step": 100,
                    "tooltip": "Maximum NEW SNAC tokens to generate per chunk (excludes input prompt tokens). Higher = longer audio per chunk (~50 tokens/word). 4000 tokens ≈ 30-40s audio"
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
                "chunk_longform": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Split long text into chunks at sentence boundaries with smooth crossfading. Enables unlimited audio length beyond the 18-20s limit"
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "audio/maya1"

    def cleanup_vram(self):
        """
        Native ComfyUI VRAM cleanup - unloads all models and clears cache.
        Follows best practices from ComfyUI's memory management system.
        """
        print("🗑️  Cleaning up VRAM...")

        # Step 1: Unload all models from VRAM
        mm.unload_all_models()

        # Step 2: Clear ComfyUI's internal cache
        mm.soft_empty_cache()

        # Step 3: Python garbage collection
        gc.collect()

        # Step 4: Clear CUDA caches (if available)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        print("✅ VRAM cleanup complete")

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
        max_new_tokens: int,
        repetition_penalty: float,
        seed: int,
        chunk_longform: bool,
        emotion_tag_insert: str = "(none)",
        chunk_index: int = None,
        total_chunks: int = None
    ) -> Tuple[dict]:
        """
        Load model (if needed) and generate expressive speech.

        Returns:
            Tuple containing audio dictionary for ComfyUI
        """
        # Import ComfyUI utilities for progress and cancellation
        import comfy.utils
        import comfy.model_management as mm

        # Check for cancellation before starting
        mm.throw_exception_if_processing_interrupted()

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

        # Check device availability
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU")
            device = "cpu"

        # ========== MODEL LOADING ==========
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

        # Strip "(BNB)" suffix from dtype labels if present
        dtype_clean = dtype.replace(" (BNB)", "")

        # Load model using the wrapper (with caching)
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

        mm.throw_exception_if_processing_interrupted()

        # ========== SPEECH GENERATION ==========
        print(f"Keep in VRAM: {keep_model_in_vram}")
        print(f"Voice: {voice_description[:60]}...")
        print(f"Text: {text[:60]}...")
        print(f"Temperature: {temperature}, Top-p: {top_p}")
        print(f"Max tokens: {max_new_tokens}")
        print("=" * 70)

        # ========== LONGFORM CHUNKING ==========
        # Check if text should be chunked (enabled + text is reasonably long)
        word_count = len(text.split())
        if chunk_longform and word_count > 80:  # Only chunk if >80 words
            print(f"📚 Longform mode enabled: {word_count} words detected")
            print(f"🔪 Splitting text into chunks at sentence boundaries...")

            # Calculate words per chunk based on max_new_tokens
            # Empirical data: 1 word ≈ 50-55 SNAC tokens
            # Leave some headroom (80%) to avoid exceeding max_new_tokens
            estimated_words_per_chunk = int((max_new_tokens * 0.8) / 50)
            estimated_words_per_chunk = max(50, min(estimated_words_per_chunk, 300))  # Clamp between 50-300

            print(f"📏 Max tokens: {max_new_tokens} → ~{estimated_words_per_chunk} words per chunk (~{estimated_words_per_chunk / 150:.1f}min per chunk)")

            text_chunks = split_text_smartly(text, max_words_per_chunk=estimated_words_per_chunk)
            print(f"📦 Split into {len(text_chunks)} chunks")
            print("=" * 70)

            # Create outer progress bar for chunks (layered progress)
            import comfy.utils
            chunk_progress = comfy.utils.ProgressBar(len(text_chunks))

            all_audio_data = []
            sample_rate = None

            for i, chunk_text in enumerate(text_chunks):
                # Create visual progress display for chunks
                chunk_bar = create_progress_bar(i + 1, len(text_chunks), width=6)
                print(f"\n🎤 Chunk Progress: {chunk_bar}")
                print(f"📝 Text: {chunk_text[:60]}...")
                print("=" * 70)

                # Check for cancellation before each chunk
                mm.throw_exception_if_processing_interrupted()

                # Recursively call generate_speech for this chunk with chunk_longform=False
                # to avoid infinite recursion
                chunk_audio = self.generate_speech(
                    model_name=model_name,
                    dtype=dtype,
                    attention_mechanism=attention_mechanism,
                    device=device,
                    voice_description=voice_description,
                    text=chunk_text,
                    keep_model_in_vram=True,  # Keep in VRAM between chunks
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=repetition_penalty,
                    seed=actual_seed,  # Use same seed for all chunks
                    chunk_longform=False,  # Disable chunking for recursive calls
                    emotion_tag_insert=emotion_tag_insert,
                    chunk_index=i + 1,  # Pass chunk context for layered progress
                    total_chunks=len(text_chunks)
                )

                # Extract audio data (returns tuple, get first element)
                chunk_audio_dict = chunk_audio[0]
                chunk_waveform = chunk_audio_dict["waveform"]
                sample_rate = chunk_audio_dict["sample_rate"]

                # Update chunk progress (outer progress bar)
                chunk_progress.update(1)
                all_audio_data.append(chunk_waveform)

                mm.throw_exception_if_processing_interrupted()

            print(f"\n{'=' * 70}")
            print(f"🔗 Combining {len(all_audio_data)} audio chunks with crossfading...")

            # Combine audio chunks with crossfading for smooth transitions
            # Crossfade duration: 50ms = 1200 samples at 24kHz
            combined_waveform_np = all_audio_data[0]

            for i in range(1, len(all_audio_data)):
                # Crossfade between chunks (1200 samples = 50ms at 24kHz)
                combined_waveform_np = crossfade_audio(
                    combined_waveform_np,
                    all_audio_data[i],
                    crossfade_samples=1200
                )

            # Ensure it's a torch tensor
            if not isinstance(combined_waveform_np, torch.Tensor):
                combined_waveform = torch.from_numpy(combined_waveform_np)
            else:
                combined_waveform = combined_waveform_np

            print(f"✅ Generated {combined_waveform.shape[-1] / sample_rate:.2f}s of audio from {len(text_chunks)} chunks")
            print("=" * 70)

            # Handle VRAM cleanup if requested
            if not keep_model_in_vram:
                print("🗑️  Offloading model from VRAM...")
                Maya1ModelLoader.clear_cache(force=True)
                print("✅ Model offloaded from VRAM")

            return ({
                "waveform": combined_waveform,
                "sample_rate": sample_rate
            },)

        # ========== SINGLE GENERATION (NO CHUNKING) ==========
        # Set seed for reproducibility
        torch.manual_seed(actual_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(actual_seed)

        # Format prompt using Maya1's OFFICIAL format (from transformers_inference.py)
        print("🔤 Formatting prompt with control tokens...")

        # Official Maya1 control token IDs
        SOH_ID = 128259  # Start of Header
        EOH_ID = 128260  # End of Header
        SOA_ID = 128261  # Start of Audio
        CODE_START_TOKEN_ID = 128257  # Start of Speech codes
        TEXT_EOT_ID = 128009  # End of Text

        # Decode control tokens
        soh_token = maya1_model.tokenizer.decode([SOH_ID])
        eoh_token = maya1_model.tokenizer.decode([EOH_ID])
        soa_token = maya1_model.tokenizer.decode([SOA_ID])
        sos_token = maya1_model.tokenizer.decode([CODE_START_TOKEN_ID])
        eot_token = maya1_model.tokenizer.decode([TEXT_EOT_ID])
        bos_token = maya1_model.tokenizer.bos_token

        # Build formatted text
        formatted_text = f'<description="{voice_description}"> {text}'

        # Construct full prompt with all control tokens (CRITICAL for avoiding garbling!)
        prompt = (
            soh_token + bos_token + formatted_text + eot_token +
            eoh_token + soa_token + sos_token
        )

        # Debug: Print formatted prompt
        print(f"📝 Formatted text: {formatted_text[:100]}...")
        print(f"📝 Full prompt preview (first 200 chars): {repr(prompt[:200])}...")

        # Tokenize input
        inputs = maya1_model.tokenizer(
            prompt,
            return_tensors="pt"
        )
        print(f"📊 Input token count: {inputs['input_ids'].shape[1]}")

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Check for cancellation
        mm.throw_exception_if_processing_interrupted()

        # Generate with progress tracking and cancellation checks
        print(f"🎵 Generating speech (max {max_new_tokens} tokens)...")

        try:
            # Setup progress tracking
            from comfy.utils import ProgressBar
            progress_bar = ProgressBar(max_new_tokens)

            # Create stopping criteria for cancellation support
            from transformers import StoppingCriteria, StoppingCriteriaList

            class InterruptionStoppingCriteria(StoppingCriteria):
                """Custom stopping criteria that checks for ComfyUI cancellation."""
                def __init__(self, progress_bar, chunk_index=None, total_chunks=None):
                    self.progress_bar = progress_bar
                    self.current_tokens = 0
                    self.input_length = 0
                    self.start_time = None
                    self.last_print_time = None
                    self.print_interval = 0.5  # Print progress every 0.5 seconds
                    self.chunk_index = chunk_index
                    self.total_chunks = total_chunks

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

                        # Print progress with visual bar and it/s to console
                        current_time = time.time()
                        if current_time - self.last_print_time >= self.print_interval:
                            elapsed = current_time - self.start_time
                            it_per_sec = new_tokens / elapsed if elapsed > 0 else 0

                            # Create visual progress bar for tokens
                            token_bar = create_progress_bar(new_tokens, max_new_tokens, width=12)

                            # Show layered progress if in chunked mode
                            if self.chunk_index is not None and self.total_chunks is not None:
                                chunk_bar = create_progress_bar(self.chunk_index, self.total_chunks, width=6, show_numbers=False)
                                print(f"   Chunk {chunk_bar} → Token Progress: {token_bar} | Speed: {it_per_sec:.2f} it/s", end='\r')
                            else:
                                print(f"   Progress: {token_bar} | Speed: {it_per_sec:.2f} it/s | Elapsed: {elapsed:.1f}s", end='\r')

                            self.last_print_time = current_time

                    # Check for cancellation using ComfyUI's native method
                    try:
                        mm.throw_exception_if_processing_interrupted()
                    except:
                        # If interrupted, stop generation gracefully
                        print("\n🛑 Generation cancelled by user")
                        return True  # Stop generation

                    return False  # Continue generation

            stopping_criteria = StoppingCriteriaList([
                InterruptionStoppingCriteria(progress_bar, chunk_index=chunk_index, total_chunks=total_chunks)
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
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=28,  # At least 4 SNAC frames (4 frames × 7 tokens = 28)
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=maya1_model.tokenizer.pad_token_id,
                    eos_token_id=128258,  # CODE_END_TOKEN_ID - Stop at end of speech
                    stopping_criteria=stopping_criteria,
                    use_cache=True,  # Enable KV cache for faster generation
                )

            generation_time = time.time() - generation_start

            # Check for cancellation after generation
            mm.throw_exception_if_processing_interrupted()

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
            mm.throw_exception_if_processing_interrupted()

            # Decode SNAC tokens to audio
            print("🔊 Decoding to audio...")
            audio_waveform = SNACDecoder.decode(snac_tokens, device=device)

            # Check for cancellation after decoding
            mm.throw_exception_if_processing_interrupted()

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
