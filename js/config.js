/**
 * Maya1 TTS Configuration
 * Tooltips, Character Presets, and Emotion Tags
 */

export const tooltips = {
    // Model settings
    model_name: "Select your downloaded Maya1 model from models/maya1-TTS/",
    dtype: "Model precision:\n• float16: Fast, 8GB VRAM, good quality\n• bfloat16: Most stable, 8GB VRAM (recommended)\n• float32: Best quality, 16GB VRAM, slower\n• 4bit/8bit: Saves VRAM but slower generation",
    attention_mechanism: "Attention implementation:\n• sdpa: Default, fast, works everywhere\n• flash_attention_2: Fastest (requires: pip install flash-attn)\n• sage_attention: Memory efficient for long sequences",
    device: "Hardware to run on:\n• cuda: Use GPU (fast, needs VRAM)\n• cpu: Use CPU only (slow, no VRAM needed)",

    // Voice and text
    voice_description: "Describe the voice characteristics:\n• Age, gender, accent\n• Tone (warm, cold, energetic)\n• Pacing (slow, conversational, fast)\n• Timbre (deep, high-pitched, raspy)\n\nExample: 'Female in 20s, British accent, warm tone, conversational pacing'\n\n💡 Ctrl+Enter to save | Escape to cancel | Click outside to save\n⏎ Enter for new line",
    text: "Your text to speak. Can include emotion tags like:\n<laugh> <whisper> <excited> <angry> <cry>\n\nClick emotion buttons below to insert tags easily!\n\n💡 Ctrl+Enter to save | Escape to cancel | Click outside to save\n⏎ Enter for new line\n⛶ Click expand button for longform text editor",

    // Generation settings
    keep_model_in_vram: "Keep model loaded after generation:\n• ON: Faster repeated generations (uses 8-16GB VRAM)\n• OFF: Clears VRAM after each generation",
    temperature: "Controls randomness (0.1-2.0):\n• 0.4: Recommended, balanced\n• Lower (0.1-0.3): More consistent, robotic\n• Higher (0.5-1.0): More creative, varied",
    top_p: "Nucleus sampling (0.1-1.0):\n• 0.9: Recommended, natural speech\n• Lower: More focused, less variety\n• Higher: More diverse but less coherent",
    max_tokens: "Maximum audio tokens to generate:\n• ~500 tokens ≈ 10 seconds\n• ~1000 tokens ≈ 20 seconds\n• ~2000 tokens ≈ 40 seconds",
    repetition_penalty: "Prevents repetitive patterns:\n• 1.1: Recommended\n• Higher (1.2-1.5): Reduces loops but may affect quality\n• 1.0: No penalty (may loop)",
    seed: "Random seed for reproducibility:\n• 0: Random output each time\n• Fixed number (e.g., 42): Same output with same inputs",
    chunk_longform: "⚠️ EXPERIMENTAL: Auto-split long text:\n• ON: Splits text >80 words at sentences, combines audio\n• OFF: Generates entire text at once (may fail if too long)",
    debug_mode: "Console output verbosity:\n• ON: Shows detailed info (token IDs, timings, stats)\n• OFF: Shows only essentials (seed, VRAM, progress)",

    // Emotion tag insert dropdown
    emotion_tag_insert: "Legacy emotion tag selector\n(Use clickable buttons below instead!)"
};

export const characterPresets = [
    {
        emoji: "♂️",
        name: "Male US",
        description: "Realistic male voice in the 30s age with american accent. Normal pitch, warm timbre, conversational pacing."
    },
    {
        emoji: "♀️",
        name: "Female UK",
        description: "Realistic female voice in the 20s age with british accent. Normal pitch, warm timbre, conversational pacing."
    },
    {
        emoji: "🎙️",
        name: "Announcer",
        description: "Professional male announcer voice in the 40s age with american accent. Rich pitch, powerful timbre, clear measured pacing."
    },
    {
        emoji: "🤖",
        name: "Robot",
        description: "Robotic AI voice, neutral gender in synthetic age. Monotone pitch, metallic timbre, precise mechanical pacing, emotionless delivery."
    },
    {
        emoji: "😈",
        name: "Demon",
        description: "Demonic entity voice, deep male in unknown age with hellish accent. Very low pitch, gravelly timbre, menacing pacing, evil tone."
    }
];

// All emotion tags use the same purple gradient color for consistency
const EMOTION_COLOR = "#667eea";  // Purple accent matching theme

export const emotionTags = [
    { tag: "<laugh>", display: "laugh", color: EMOTION_COLOR },
    { tag: "<laugh_harder>", display: "laugh harder", color: EMOTION_COLOR },
    { tag: "<chuckle>", display: "chuckle", color: EMOTION_COLOR },
    { tag: "<giggle>", display: "giggle", color: EMOTION_COLOR },
    { tag: "<sigh>", display: "sigh", color: EMOTION_COLOR },
    { tag: "<gasp>", display: "gasp", color: EMOTION_COLOR },
    { tag: "<angry>", display: "angry", color: EMOTION_COLOR },
    { tag: "<excited>", display: "excited", color: EMOTION_COLOR },
    { tag: "<whisper>", display: "whisper", color: EMOTION_COLOR },
    { tag: "<cry>", display: "cry", color: EMOTION_COLOR },
    { tag: "<scream>", display: "scream", color: EMOTION_COLOR },
    { tag: "<sing>", display: "sing", color: EMOTION_COLOR },
    { tag: "<snort>", display: "snort", color: EMOTION_COLOR },
    { tag: "<exhale>", display: "exhale", color: EMOTION_COLOR },
    { tag: "<gulp>", display: "gulp", color: EMOTION_COLOR },
    { tag: "<sarcastic>", display: "sarcastic", color: EMOTION_COLOR }
];
