"""
SNAC (Speech Neural Audio Codec) decoder for Maya1 TTS.
Handles unpacking 7-token frames and decoding to 24kHz audio.
"""

import torch
import numpy as np
from typing import List, Tuple


# Maya1 SNAC token range: 128266 to 156937
SNAC_TOKEN_START = 128266
SNAC_TOKEN_END = 156937
SNAC_CODEBOOK_SIZE = 4096  # Each level uses 4096 codes


def is_snac_token(token_id: int) -> bool:
    """
    Check if a token ID is a SNAC audio token.

    Args:
        token_id: Token ID to check

    Returns:
        True if the token is a SNAC token
    """
    return SNAC_TOKEN_START <= token_id <= SNAC_TOKEN_END


def filter_snac_tokens(token_ids: List[int]) -> List[int]:
    """
    Filter only SNAC tokens from a list of token IDs.

    Args:
        token_ids: List of token IDs (may include text tokens)

    Returns:
        List of only SNAC tokens
    """
    return [t for t in token_ids if is_snac_token(t)]


def unpack_snac_tokens(snac_tokens: List[int]) -> Tuple[List[List[int]], int]:
    """
    Unpack 7-token SNAC frames into 3 hierarchical codebook levels.

    Maya1 packs SNAC codes into 7 tokens per frame:
    - Frame: [slot0, slot1, slot2, slot3, slot4, slot5, slot6]
    - L1 (12Hz): slot0
    - L2 (23Hz): slot1, slot4
    - L3 (47Hz): slot2, slot3, slot5, slot6

    Args:
        snac_tokens: List of SNAC token IDs (should be multiple of 7)

    Returns:
        Tuple of (codes, num_frames):
        - codes: List of 3 lists [L1, L2, L3] with unpacked codes
        - num_frames: Number of frames processed
    """
    num_frames = len(snac_tokens) // 7

    if len(snac_tokens) % 7 != 0:
        print(f"⚠️  Warning: SNAC tokens ({len(snac_tokens)}) not divisible by 7. "
              f"Truncating to {num_frames * 7} tokens.")

    # Initialize codebook levels
    l1_codes = []  # 1 code per frame (12 Hz)
    l2_codes = []  # 2 codes per frame (23 Hz)
    l3_codes = []  # 4 codes per frame (47 Hz)

    for i in range(num_frames):
        # Extract 7 tokens for this frame
        frame_start = i * 7
        slots = snac_tokens[frame_start:frame_start + 7]

        # Unpack to codebook indices (subtract offset and mod by codebook size)
        l1_codes.append((slots[0] - SNAC_TOKEN_START) % SNAC_CODEBOOK_SIZE)

        l2_codes.append((slots[1] - SNAC_TOKEN_START) % SNAC_CODEBOOK_SIZE)
        l2_codes.append((slots[4] - SNAC_TOKEN_START) % SNAC_CODEBOOK_SIZE)

        l3_codes.append((slots[2] - SNAC_TOKEN_START) % SNAC_CODEBOOK_SIZE)
        l3_codes.append((slots[3] - SNAC_TOKEN_START) % SNAC_CODEBOOK_SIZE)
        l3_codes.append((slots[5] - SNAC_TOKEN_START) % SNAC_CODEBOOK_SIZE)
        l3_codes.append((slots[6] - SNAC_TOKEN_START) % SNAC_CODEBOOK_SIZE)

    codes = [l1_codes, l2_codes, l3_codes]
    return codes, num_frames


def decode_snac_to_audio(codes: List[List[int]], snac_model, device: str = "cuda") -> np.ndarray:
    """
    Decode SNAC codes to audio waveform using the SNAC decoder.

    Args:
        codes: List of 3 lists [L1, L2, L3] with unpacked codes
        snac_model: Loaded SNAC model with decoder
        device: Device to run decoding on

    Returns:
        Audio waveform as numpy array (24kHz, mono, float32)
    """
    # Convert codes to tensors
    codes_tensor = [
        torch.tensor(level_codes, dtype=torch.long, device=device).unsqueeze(0)
        for level_codes in codes
    ]

    # Decode using SNAC quantizer + decoder
    with torch.inference_mode():
        quantized = snac_model.quantizer.from_codes(codes_tensor)
        audio_tensor = snac_model.decoder(quantized)

    # Extract audio: shape is [batch, channels, samples]
    audio = audio_tensor[0, 0].cpu().numpy()

    # Trim warmup samples (first 2048 samples) - from official transformers_inference.py
    if len(audio) > 2048:
        audio = audio[2048:]

    return audio


class SNACDecoder:
    """
    Wrapper class for SNAC decoding with model caching.
    """

    _cached_model = None
    _cached_device = None

    @classmethod
    def load_snac_model(cls, device: str = "cuda"):
        """
        Load SNAC 24kHz model with caching.

        Args:
            device: Device to load model on

        Returns:
            Loaded SNAC model
        """
        # Return cached model if available
        if cls._cached_model is not None and cls._cached_device == device:
            return cls._cached_model

        print("📦 Loading SNAC 24kHz decoder...")

        try:
            from snac import SNAC
        except ImportError:
            raise ImportError(
                "SNAC package not found. Install with: pip install snac\n"
                "GitHub: https://github.com/hubertsiuzdak/snac"
            )

        # Load SNAC 24kHz model
        snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)

        # Cache the model
        cls._cached_model = snac_model
        cls._cached_device = device

        print(f"✅ SNAC decoder loaded on {device}")

        return snac_model

    @classmethod
    def decode(cls, snac_tokens: List[int], device: str = "cuda") -> np.ndarray:
        """
        Full pipeline: filter tokens → unpack → decode to audio.

        Args:
            snac_tokens: List of SNAC token IDs
            device: Device to run on

        Returns:
            Audio waveform as numpy array (24kHz, mono, float32)
        """
        # Load SNAC model (cached)
        snac_model = cls.load_snac_model(device)

        # Unpack tokens to codes
        codes, num_frames = unpack_snac_tokens(snac_tokens)

        if num_frames == 0:
            print("⚠️  No SNAC frames to decode!")
            return np.zeros(0, dtype=np.float32)

        print(f"🎵 Decoding {num_frames} SNAC frames (~{num_frames * 0.021:.2f}s audio)...")

        # Decode to audio
        audio = decode_snac_to_audio(codes, snac_model, device)

        return audio
