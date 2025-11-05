"""
ComfyUI-Maya1_TTS: Maya1 Text-to-Speech Integration for ComfyUI

Maya1 is a 3B-parameter speech model built for expressive voice generation
with rich human emotion and precise voice design.

Features:
- Voice design through natural language descriptions
- 20+ emotions: laugh, cry, whisper, angry, sigh, gasp, and more
- Real-time streaming with SNAC neural codec
- Multiple attention mechanisms: SDPA, Flash Attention 2, Sage Attention
- Native ComfyUI cancel support

Author: Maya Research
License: Apache 2.0
"""

import os
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .core.model_wrapper import Maya1ModelLoader

__version__ = "1.0.0"

# ComfyUI requires these exports
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# Tell ComfyUI where to find our JavaScript extensions
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")

# Note: VRAM management is controlled by the keep_model_in_vram toggle in the node
# Maya1 models are kept in a separate cache and are not affected by ComfyUI's
# "Unload Models" button. Use the toggle in the node to control VRAM usage.

# Print banner on load
print("=" * 70)
print("🎤 ComfyUI-Maya1_TTS")
print("   Expressive Voice Generation with Emotions")
print("=" * 70)
print("📦 Nodes loaded:")
for node_name in NODE_CLASS_MAPPINGS.keys():
    display_name = NODE_DISPLAY_NAME_MAPPINGS.get(node_name, node_name)
    print(f"   • {display_name} ({node_name})")
print("=" * 70)
