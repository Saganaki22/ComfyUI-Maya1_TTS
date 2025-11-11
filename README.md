# ComfyUI-Maya1_TTS

**Expressive Voice Generation with Emotions for ComfyUI**

A ComfyUI node pack for [Maya1](https://huggingface.co/maya-research/maya1), a 3B-parameter speech model built for expressive voice generation with rich human emotion and precise voice design.

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![ComfyUI](https://img.shields.io/badge/ComfyUI-compatible-green.svg)

https://github.com/user-attachments/assets/1be0c2a0-22fb-4890-9147-d20abeb2e067


---

## ✨ Features

### Core Features
- 🎭 **Voice Design** through natural language descriptions
- 😊 **16 Emotion Tags**: laugh, cry, whisper, angry, sigh, gasp, scream, and more
- ⚡ **Real-time Generation** with SNAC neural codec (24kHz audio)
- 🔧 **Multiple Attention Mechanisms**: SDPA, eager, Flash Attention 2, Sage Attention (1/2)
- 💾 **Quantization Support**: 4-bit and 8-bit for memory-constrained GPUs
- 🛑 **Native ComfyUI Cancel**: Stop generation anytime
- 📊 **Progress Tracking**: Real-time token generation speed (it/s)
- 🔄 **Model Caching**: Fast subsequent generations
- 🎯 **Smart VRAM Management**: Auto-clears on dtype changes

### Custom Canvas UI
- 🎨 **Beautiful Dark Theme** with purple accents and smooth animations
- 👤 **5 Character Presets**: Quick-load voice templates (Male US, Female UK, Announcer, Robot, Demon)
- 🎭 **16 Visual Emotion Buttons**: One-click emotion tag insertion at cursor position
- ⛶ **Professional HTML Modal Editor**: Fullscreen text editor with native textarea for longform content
- 🔤 **Font Size Controls**: Adjustable 12-20px font size with visual slider
- ⌨️ **Advanced Keyboard Shortcuts**: Ctrl+A, Ctrl+C, Ctrl+V, Ctrl+X, Ctrl+Enter to save, ESC to cancel
- 🔔 **Toast Notifications**: Visual feedback for save success and validation errors
- 📝 **Inline Text Editing**: Click-to-edit with cursor positioning and drag-to-select
- 🖱️ **Scroll Support**: Custom themed scrollbars with mouse wheel scrolling
- 📱 **Responsive Design**: Modal adapts to all screen sizes
- 💡 **Contextual Tooltips**: Helpful hints on every control
- 🎬 **Collapsible Sections**: Clean, organized interface
- 🔄 **Smart Audio Processing**: Auto-chunking for long text with crossfade blending for seamless output

---

## 📦 Installation

<details>
<summary><b>Quick Install (Click to expand)</b></summary>

### 1. Clone the Repository

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/Saganaki22/ComfyUI-Maya1_TTS.git
cd ComfyUI-Maya1_TTS
```

### 2. Install Dependencies

**Core dependencies** (required):
```bash
pip install torch>=2.0.0 transformers>=4.50.0 numpy>=1.21.0 snac>=1.0.0
```

**Or install from requirements.txt:**
```bash
pip install -r requirements.txt
```

</details>

<details>
<summary><b>Optional: Enhanced Performance (Click to expand)</b></summary>

### Quantization (Memory Savings)

For 4-bit/8-bit quantization support:
```bash
pip install bitsandbytes>=0.41.0
```

**Memory savings:**
- 4-bit: ~6GB → (slight quality loss)
- 8-bit: ~6GB → (minimal quality loss)

### Accelerated Attention

**Flash Attention 2** (CUDA only):
```bash
pip install flash-attn>=2.0.0
```

**Sage Attention** (memory efficient for batch):
```bash
pip install sageattention>=1.0.0
```

### Install All Optional Dependencies

```bash
pip install bitsandbytes flash-attn sageattention
```

</details>

<details>
<summary><b>Download Maya1 Model (Click to expand)</b></summary>

### Model Location

Models go in: `ComfyUI/models/maya1-TTS/`

### Expected Folder Structure

After downloading, your model folder should look like this:

```
ComfyUI/
└── models/
    └── maya1-TTS/
        └── maya1/                                # Model name (can be anything)
            ├── chat_template.jinja               # Chat template
            ├── config.json                       # Model configuration
            ├── generation_config.json            # Generation settings
            ├── model-00001-of-00002.safetensors  # Model weights (shard 1)
            ├── model-00002-of-00002.safetensors  # Model weights (shard 2)
            ├── model.safetensors.index.json      # Weight index
            ├── special_tokens_map.json           # Special tokens
            └── tokenizer/                        # Tokenizer subfolder
                ├── chat_template.jinja           # Chat template (duplicate)
                ├── special_tokens_map.json       # Special tokens (duplicate)
                ├── tokenizer.json                # Tokenizer vocabulary (22.9 MB)
                └── tokenizer_config.json         # Tokenizer config
```

**Critical files required:**
- `config.json` - Model architecture configuration
- `generation_config.json` - Default generation parameters
- `model-00001-of-00002.safetensors` & `model-00002-of-00002.safetensors` - Model weights (2 shards)
- `model.safetensors.index.json` - Weight index mapping
- `chat_template.jinja` & `special_tokens_map.json` - In root folder
- `tokenizer/` folder with all 4 tokenizer files

**Note:** You can have multiple models by creating separate folders like `maya1`, `maya1-finetuned`, etc.

### Option 1: Hugging Face CLI (Recommended)

```bash
# Install HF CLI
pip install huggingface-hub

# Create directory
cd ComfyUI
mkdir -p models/maya1-TTS

# Download model
hf download maya-research/maya1 --local-dir models/maya1-TTS/maya1
```

### Option 2: Python Script

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="maya-research/maya1",
    local_dir="ComfyUI/models/maya1-TTS/maya1",
    local_dir_use_symlinks=False
)
```

### Option 3: Manual Download

1. Go to [Maya1 on HuggingFace](https://huggingface.co/maya-research/maya1)
2. Download all files to `ComfyUI/models/maya1-TTS/maya1/`

</details>

<details>
<summary><b>Restart ComfyUI</b></summary>

Restart ComfyUI to load the new nodes. The node will appear under:

**Add Node → audio → Maya1 TTS (AIO) / Maya1 TTS (AIO) Barebones**

</details>

---

## 🎮 Usage

### Two Node Options

**Maya1 TTS (AIO)** - Full custom UI with visual controls (recommended)
- Beautiful dark theme with character presets, emotion buttons, and modal editor
- Best user experience with visual feedback and tooltips

**Maya1 TTS (AIO) Barebones** - Standard ComfyUI widgets only
- For users experiencing JavaScript rendering issues (black box)
- Same functionality, simpler interface
- All inputs stacked vertically with standard dropdowns and text boxes

---

### Node: Maya1 TTS (AIO)

All-in-one node for loading models and generating speech with a beautiful custom canvas UI.

| Maya1 TTS (AIO) | Maya1 TTS (AIO) Barebones |
|:---:|:---:|
| <img width="615" alt="Screenshot 2025-11-07 084153" src="https://github.com/user-attachments/assets/19105cc2-030a-40e3-b4d9-e18bd6d50b65" /> | <img width="648" alt="image" src="https://github.com/user-attachments/assets/7aff4fbb-c434-4d16-b4f9-22017167d455" /> |



### ✨ Custom Canvas Interface

The node features a completely custom-built interface with:

**Character Presets** (Top Row)
- Click any preset to instantly load a pre-configured voice description
- 5 presets: ♂️ Male US, ♀️ Female UK, 🎙️ Announcer, 🤖 Robot, 😈 Demon

**Text Fields**
- **Voice Description**: Describe your desired voice characteristics
- **Text**: Your script with optional emotion tags
- Click inside to edit with full keyboard support
- Press **Enter** for new line, **Ctrl+Enter** to save, **Escape** to cancel

**Emotion Tags** (Collapsible Grid)
- 16 emotion buttons in 4×4 grid
- Click any emotion to insert tag at cursor position
- Tags insert where you're typing, not just at the end
- Click header to collapse/expand section

**⛶ Professional HTML Modal** (Bottom right of Text field)
- Click the expand button (⛶) for fullscreen text editing
- Native HTML textarea with proper newline and whitespace support
- **Font Size Slider**: Adjust text size from 12px to 20px with visual A/A controls
- All 16 emotion buttons available inside modal for quick tag insertion
- **Custom Themed Scrollbar**: Purple accents matching the node design
- **Toast Notifications**: Green checkmark for "Text Saved", red X for validation errors
- **Empty Text Validation**: Prevents saving blank text with helpful error message
- **Keyboard Shortcuts**:
  - **Ctrl+Enter**: Save and close
  - **ESC**: Cancel without saving
  - Full text selection and clipboard support (Ctrl+A, C, V, X)
- **Responsive Design**: Modal adapts to small and large screens, buttons always visible
- **Visual Hints**: Subtle grey text under buttons showing keyboard shortcuts

**Keyboard Shortcuts** (Inline Editing & Modal)
- `Enter`: New line (in multiline text fields)
- `Ctrl+Enter`: Save and apply changes
- `Escape`: Cancel editing without saving
- `Ctrl+A`: Select all text
- `Ctrl+C/V/X`: Copy, paste, cut selected text
- Click outside field: Auto-save (inline editing only)

<details>
<summary><b>Model Settings</b></summary>

**model_name** (dropdown)
- Select from models in `ComfyUI/models/maya1-TTS/`
- Model auto-discovered on startup

**dtype** (dropdown)
- `4bit`: NF4 quantization (~6GB VRAM, requires bitsandbytes, **SLOWER**)
- `8bit`: INT8 quantization (~7GB VRAM, requires bitsandbytes, **SLOWER**)
- `float16`: 16-bit half precision (~8-9GB VRAM, **FAST**, good quality)
- `bfloat16`: 16-bit brain float (~8-9GB VRAM, **FAST**, recommended)
- `float32`: 32-bit full precision (~16GB VRAM, highest quality, slower)

⚠️ **IMPORTANT:** Quantization (4-bit/8-bit) is **SLOWER** than float16/bfloat16!
- Only use quantization if you have **limited VRAM** (<10GB)
- If you have **10GB+ VRAM**, use **float16** or **bfloat16** for best speed

**attention_mechanism** (dropdown)
- `sdpa`: PyTorch SDPA (**default**, fastest for single TTS)
- `flash_attention_2`: Flash Attention 2 (batch inference)
- `sage_attention`: Sage Attention (memory efficient)

**device** (dropdown)
- `cuda`: Use GPU (recommended)
- `cpu`: Use CPU (slower)

</details>

<details>
<summary><b>Voice & Text Settings</b></summary>

**voice_description**

Describe the voice using natural language. Click inside to edit or use character presets.

**Example:**
```
Realistic male voice in the 30s with American accent. Normal pitch, warm timbre, conversational pacing.
```

**Voice Components:**
- **Age**: `in their 20s`, `30s`, `40s`, `50s`
- **Gender**: `Male voice`, `Female voice`
- **Accent**: `American`, `British`, `Australian`, `Indian`, `Middle Eastern`
- **Pitch**: `high pitch`, `normal pitch`, `low pitch`
- **Timbre**: `warm`, `gravelly`, `smooth`, `raspy`
- **Pacing**: `fast pacing`, `conversational`, `slow pacing`
- **Tone**: `happy`, `angry`, `curious`, `energetic`, `calm`

**💡 Tip**: Use character presets for quick voice templates!

**text**

Text to synthesize with optional emotion tags. Click emotion buttons to insert tags at cursor.

**Example:**
```
Hello! This is Maya1 <laugh> the best open source voice AI!
```

**💡 Tip**: Click ⛶ expand button for longform text editing in fullscreen modal!

</details>

<details>
<summary><b>Generation Settings</b></summary>

**keep_model_in_vram** (boolean)
- `True`: Keep model loaded for faster repeated generations
- `False`: Clear VRAM after generation (saves memory)
- Auto-clears when dtype changes

**chunk_longform** (boolean) ⚠️ EXPERIMENTAL
- `True`: Auto-split long text (>80 words) at sentences, combines audio
- `False`: Generate entire text at once (may fail if too long)
- **Note**: This feature is experimental and may have quality/timing issues

**temperature** (0.1-2.0, default: 0.4)
- Lower = more consistent
- Higher = more varied/creative

**top_p** (0.1-1.0, default: 0.9)
- Nucleus sampling parameter
- 0.9 recommended for natural speech

**max_tokens** (100-8000, default: 2000)
- Maximum audio tokens to generate
- Higher = longer audio

**repetition_penalty** (1.0-2.0, default: 1.1)
- Reduces repetitive speech
- 1.1 is good default

**seed** (integer, default: 0)
- Use same seed for reproducible results
- Use ComfyUI's control_after_generate for random/increment

</details>

<details>
<summary><b>Outputs</b></summary>

**audio** (ComfyUI AUDIO type)
- 24kHz mono audio
- Compatible with all ComfyUI audio nodes
- Connect to PreviewAudio, SaveAudio, etc.

</details>

---

### Node: Maya1 TTS (AIO) Barebones

Standard ComfyUI widgets version for users experiencing JavaScript rendering issues.

**When to use Barebones:**
- Custom UI shows as a black box
- Browser console shows JavaScript errors
- You prefer simple, standard ComfyUI widgets
- Working with older ComfyUI versions

**Inputs (in order):**

1. **voice_description** (multiline text)
   - Describe voice characteristics in natural language
   - Same as main node, just standard text box

2. **text** (multiline text)
   - Your script with manual emotion tags like `<laugh>` or `<cry>`
   - Type emotion tags manually (no visual buttons in barebones version)

3. **model_name** (dropdown)
   - Select Maya1 model from `ComfyUI/models/maya1-TTS/`

4. **dtype** (dropdown)
   - `4bit (BNB)`, `8bit (BNB)`, `float16`, `bfloat16`, `float32`

5. **attention_mechanism** (dropdown)
   - `sdpa` (default), `flash_attention_2`, `sage_attention`

6. **device** (dropdown)
   - `cuda` (GPU) or `cpu`

7. **keep_model_in_vram** (boolean toggle)
   - Keep model loaded for faster subsequent generations

8. **chunk_longform** (boolean toggle)
   - Split long text with crossfading for unlimited length

9. **max_tokens** (integer)
   - Max SNAC tokens per chunk (default: 4000)

10. **temperature** (float)
    - Generation randomness (default: 0.4)

11. **top_p** (float)
    - Nucleus sampling (default: 0.9)

12. **repetition_penalty** (float)
    - Reduce repetition (default: 1.1)

13. **seed** (integer)
    - 0 = random, or set specific seed for reproducibility
    - Use control_after_generate widget for seed management

**All other features (model loading, VRAM management, chunking, progress tracking) work identically to the main node.**

---

## 🎭 Emotion Tags

Add emotions anywhere in your text using `<tag>` syntax, or click the visual emotion buttons in the UI!

**Examples:**
```
Hello! This is amazing <laugh> I can't believe it!
```

```
After all we went through <cry> I can't believe he was the traitor.
```

```
Wow! <gasp> This place looks incredible!
```

<details>
<summary><b>All 16 Available Emotions (Click to expand)</b></summary>

**Laughter & Joy:**
- `<laugh>` - Normal laugh
- `<laugh_harder>` - Intense laughing
- `<giggle>` - Light giggling
- `<chuckle>` - Soft chuckle

**Sadness & Sighs:**
- `<cry>` - Crying
- `<sigh>` - Sighing

**Surprise & Breath:**
- `<gasp>` - Surprised gasp
- `<excited>` - Excited tone

**Intensity & Emotion:**
- `<whisper>` - Whispering
- `<angry>` - Angry tone
- `<scream>` - Screaming
- `<sarcastic>` - Sarcastic delivery

**Natural Sounds:**
- `<snort>` - Snorting
- `<exhale>` - Exhaling
- `<gulp>` - Gulping
- `<sing>` - Singing

</details>

**💡 Tip:** Click emotion buttons in the node UI to insert tags at cursor position!

---

## 🎬 Example Character Speeches

<details>
<summary><b>Generative AI & ComfyUI Examples (Click to expand)</b></summary>

### Example 1: Excited AI Researcher

**Voice Description:**
```
Female voice in her 30s with American accent. High pitch, energetic tone at high intensity, fast pacing.
```

**Text:**
```
Oh my god! <laugh> Have you seen the new Stable Diffusion model in ComfyUI? The quality is absolutely incredible! <gasp> I just generated a photorealistic portrait in like 20 seconds. This is game-changing for our workflow!
```

---

### Example 2: Skeptical Developer

**Voice Description:**
```
Male voice in his 40s with British accent. Low pitch, calm tone, conversational pacing.
```

**Text:**
```
I've been testing this new node pack in ComfyUI <sigh> and honestly, I'm impressed. At first I was skeptical about the whole generative AI hype, but <gasp> the control you get with custom nodes is remarkable. This changes everything.
```

---

### Example 3: Enthusiastic Tutorial Creator

**Voice Description:**
```
Female voice in her 20s with Australian accent. Normal pitch, warm timbre, energetic tone at medium intensity.
```

**Text:**
```
Hey everyone! <laugh> Welcome back to my ComfyUI tutorial series! Today we're diving into the most powerful image generation workflow I've ever seen. <gasp> You're not gonna believe how easy this is! Let's get started!
```

---

### Example 4: Frustrated Beginner

**Voice Description:**
```
Male voice in his 30s with American accent. Normal pitch, stressed tone at medium intensity, fast pacing.
```

**Text:**
```
Why won't this workflow run? <angry> I've connected all the nodes exactly like the tutorial showed! <sigh> Wait... Oh no. <laugh> I forgot to load the checkpoint model. Classic beginner mistake! Okay, let's try this again.
```

---

### Example 5: Amazed AI Artist

**Voice Description:**
```
Female voice in her 40s with Indian accent. Normal pitch, curious tone, slow pacing, dramatic delivery.
```

**Text:**
```
When I first discovered ComfyUI <whisper> I thought it was just another image generator. But then <gasp> I realized you can chain workflows together, use custom models, and <laugh> even generate animations! This is the future of digital art!
```

---

### Example 6: Confident AI Entrepreneur

**Voice Description:**
```
Male voice in his 50s with Middle Eastern accent. Low pitch, gravelly timbre, slow pacing, confident tone at high intensity.
```

**Text:**
```
The generative AI revolution is here. <dramatic pause> ComfyUI gives us the tools to build production-ready workflows. <chuckle> While others are still playing with web UIs, we're automating entire creative pipelines. This is how you stay ahead of the curve.
```

</details>

---

## ⚙️ Advanced Configuration

<details>
<summary><b>Attention Mechanisms Comparison</b></summary>

| Mechanism | Speed | Memory | Best For | Requirements |
|-----------|-------|--------|----------|--------------|
| **SDPA** | ⚡⚡⚡ | Good | Single TTS generation | PyTorch ≥2.0 |
| **Flash Attention 2** | ⚡⚡ | Good | Batch processing | flash-attn, CUDA |
| **Sage Attention** | ⚡⚡ | Excellent | Long sequences | sageattention |

**Why is SDPA fastest for TTS?**
- Optimized for single-sequence autoregressive generation
- Lower kernel launch overhead (~20μs vs 50-60μs)
- Flash/Sage Attention shine with batch size ≥8

**Recommendation:** Use **SDPA** (default) for single audio generation.

</details>

<details>
<summary><b>Quantization Details</b></summary>

⚠️ **CRITICAL: Quantization is SLOWER than fp16/bf16!**

### Memory Usage (Maya1 3B Model)

| Dtype | VRAM Usage | Speed | Quality |
|-------|------------|-------|---------|
| **4-bit NF4** | ~6GB | Slow ⚡ | Good (slight loss) |
| **8-bit INT8** | ~7GB | Slow ⚡ | Excellent (minimal loss) |
| **float16** | ~8-9GB | **Fast** ⚡⚡⚡ | Excellent |
| **bfloat16** | ~8-9GB | **Fast** ⚡⚡⚡ | Excellent |
| **float32** | ~16GB | Medium ⚡⚡ | Perfect |

### 4-bit NF4 Quantization

**Features:**
- Uses NormalFloat4 (NF4) for best 4-bit quality
- Double quantization (nested) for better accuracy
- Memory savings: ~6GB (vs ~8-9GB for fp16)

**When to use:**
- You have **limited VRAM** (8GB or less GPU)
- Speed is not critical (inference is slower due to dequantization)
- Need to fit model in smaller VRAM

**When NOT to use:**
- You have 10GB+ VRAM → Use float16/bfloat16 instead for better speed!

### 8-bit INT8 Quantization

**Features:**
- Standard 8-bit integer quantization
- Memory savings: ~7GB (vs ~8-9GB for fp16)
- Minimal quality impact

**When to use:**
- You have moderate VRAM constraints (8-10GB GPU)
- Want good quality with some memory savings
- Speed is not critical

**When NOT to use:**
- You have 10GB+ VRAM → Use float16/bfloat16 instead for better speed!

### Why is Quantization Slower?

Quantized models require **dequantization** on every forward pass:
1. Model weights stored in 4-bit/8-bit
2. Weights dequantized to fp16 for computation
3. Computation happens in fp16
4. Extra overhead = slower inference

**Recommendation:** Only use quantization if you truly need the memory savings!

### Automatic Dtype Switching

The node automatically clears VRAM when you switch dtypes:

```
🔄 Dtype changed from bfloat16 to 4bit
   Clearing cache to reload model...
```

This prevents dtype mismatch errors and ensures correct quantization.

</details>

<details>
<summary><b>Console Progress Output</b></summary>

Real-time generation statistics in the console:

```
🎲 Seed: 1337
🎵 Generating speech (max 2000 tokens)...
   Tokens: 500/2000 | Speed: 12.45 it/s | Elapsed: 40.2s
✅ Generated 1500 tokens in 120.34s (12.47 it/s)
```

**it/s** = iterations per second (tokens/second)

</details>

---

## 🐛 Troubleshooting

<details>
<summary><b>Node Shows as Black Box (JavaScript Issues)</b></summary>

**Issue:** Maya1 TTS (AIO) node appears completely black with no widgets visible.

**Quick Fix:**
Use **Maya1 TTS (AIO) Barebones** instead!
- Same functionality, standard ComfyUI widgets only
- No custom JavaScript required
- Find it under: Add Node → audio → Maya1 TTS (AIO) Barebones

**Debugging Steps:**
1. Open browser DevTools (F12) → Console tab
2. Look for JavaScript errors mentioning "maya1" or "Unexpected token"
3. Try hard refresh: Ctrl+Shift+R (Windows/Linux) or Cmd+Shift+R (Mac)
4. Clear browser cache completely
5. Test in incognito/private window
6. Check if maya1_tts.js loads in Network tab (should be 200 status)
7. Disable browser extensions (ad blockers, script blockers)
8. Update ComfyUI to latest version

**Note:** The barebones version is specifically designed for this issue!

</details>

<details>
<summary><b>Model Not Found</b></summary>

**Error:** `No valid Maya1 models found`

**Solutions:**
1. Check model location: `ComfyUI/models/maya1-TTS/`
2. Download model (see Installation section)
3. Restart ComfyUI
4. Check console for model discovery messages

</details>

<details>
<summary><b>Out of Memory (OOM)</b></summary>

**Error:** `CUDA out of memory`

**Memory requirements:**
- 4-bit: ~6GB VRAM (slower)
- 8-bit: ~7GB VRAM (slower)
- float16/bfloat16: ~8-9GB VRAM (fast, recommended)
- float32: ~16GB VRAM

**Solutions (try in order):**
1. Use **4-bit** dtype if you have ≤8GB VRAM (~6GB usage)
2. Use **8-bit** dtype if you have ~8-10GB VRAM (~7GB usage)
3. Use **float16** if you have 10GB+ VRAM (faster than quantization!)
4. Enable `keep_model_in_vram=False` to free VRAM after generation
5. Reduce `max_tokens` to 1000-1500
6. Close other VRAM-heavy applications
7. Use CPU (much slower but works)

**Note:** If you have 10GB+ VRAM, use float16/bfloat16 for best speed!

</details>

<details>
<parameter name="summary"><b>Quantization Errors</b></summary>

**Error:** `bitsandbytes not found`

**Solution:**
```bash
pip install bitsandbytes>=0.41.0
```

**Error:** `Quantization requires CUDA`

**Solution:**
- 4-bit/8-bit only work on CUDA
- Switch to `float16`/`bfloat16` for CPU

</details>

<details>
<summary><b>No Audio Generated</b></summary>

**Error:** `No SNAC audio tokens generated!`

**Solutions:**
1. Increase `max_tokens` to 2000-4000
2. Adjust `temperature` to 0.3-0.5
3. Simplify voice description
4. Check text isn't too long
5. Try different seed value

</details>

<details>
<summary><b>Flash Attention Installation Failed</b></summary>

**Error:** `flash-attn` won't install

**Solution:**
- Flash Attention requires CUDA and specific setup
- Just use **SDPA** instead (works great, actually faster for TTS!)
- SDPA is the recommended default

</details>

<details>
<summary><b>Info Button Not Visible</b></summary>

**Issue:** Can't see the "?" or "i" icon, only hover tooltip

**Answer:** This is **normal** and working correctly!

- ComfyUI's `DESCRIPTION` creates a hover tooltip
- Some ComfyUI versions show no visible icon
- Just hover over the node title area to see help
- Contains all emotion tags and usage examples

</details>

---

## 📊 Performance Tips

1. **Use float16/bfloat16** if you have 10GB+ VRAM (fastest!)
2. **Use quantization (4-bit/8-bit)** ONLY if limited VRAM (<10GB) - slower but fits in memory
3. **Keep SDPA** as attention mechanism (fastest for single TTS)
4. **Enable model caching** (`keep_model_in_vram=True`) for multiple generations
5. **Optimize max_tokens**: Start with 1500-2000
6. **Batch similar requests** with same voice description for efficiency

⚠️ **Speed ranking:** float16/bfloat16 (fastest) > float32 > 8-bit > 4-bit (slowest)

---

## 🏗️ Technical Details

<details>
<summary><b>Architecture</b></summary>

- **Model**: 3B-parameter Llama-based transformer
- **Audio Codec**: SNAC (Speech Neural Audio Codec)
- **Sample Rate**: 24kHz mono
- **Frame Structure**: 7 tokens per frame (3 hierarchical levels)
- **Token Ranges**:
  - SNAC tokens: 128266-156937
  - Text EOS: 128009
  - SNAC EOS: 128258
- **Compression**: ~0.98 kbps streaming

</details>

<details>
<summary><b>File Structure</b></summary>

```
ComfyUI-Maya1_TTS/
├── __init__.py                 # Node registration
├── nodes/
│   ├── __init__.py
│   └── maya1_tts_combined.py   # AIO node (backend)
├── js/
│   ├── maya1_tts.js            # Custom canvas UI (1800+ lines)
│   └── config.js               # UI config (presets, emotions, tooltips)
├── core/
│   ├── model_wrapper.py        # Model loading & quantization
│   ├── snac_decoder.py         # SNAC audio decoding
│   └── utils.py                # Utilities & cancel support
├── resources/
│   ├── emotions.txt            # 16 emotion tags
│   └── prompt_examples.txt     # Voice description examples
├── pyproject.toml              # Package metadata
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

</details>

<details>
<summary><b>ComfyUI Integration</b></summary>

- **Custom Canvas UI**: Full JavaScript UI with LiteGraph.js canvas API
- **Cancel Support**: Native `execution.interruption_requested()`
- **Progress Bars**: `comfy.utils.ProgressBar`
- **Audio Format**: ComfyUI AUDIO type (24kHz mono)
- **Model Caching**: Automatic with dtype change detection
- **VRAM Management**: Manual control via toggle
- **Event Handling**: Document-level keyboard/mouse capture for proper text editing
- **Visual Feedback**: Real-time tooltips, animations, and hover states

</details>




---

## 📝 Credits

- **Maya1 Model**: [Maya Research](https://www.mayaresearch.ai/)
- **HuggingFace**: [maya-research/maya1](https://huggingface.co/maya-research/maya1)
- **SNAC Codec**: [hubertsiuzdak/snac](https://github.com/hubertsiuzdak/snac)
- **ComfyUI**: [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)

---

## 📄 License

Apache 2.0 - See [LICENSE](LICENSE)

Maya1 model is also licensed under Apache 2.0 by Maya Research.

---

## 🔗 Links

- **Issues**: [GitHub Issues](https://github.com/Saganaki22/-ComfyUI-Maya1_TTS/issues)
- **Maya Research**: [Website](https://www.mayaresearch.ai/) | [Twitter](https://twitter.com/mayaresearch_ai)
- **Model Page**: [HuggingFace](https://huggingface.co/maya-research/maya1)

---

## 📖 Citation

If you use Maya1 in your research, please cite:

```bibtex
@misc{maya1voice2025,
  title={Maya1: Open Source Voice AI with Emotional Intelligence},
  author={Maya Research},
  year={2025},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/maya-research/maya1}},
}
```

---

*Bringing expressive voice AI to everyone through open source.*
