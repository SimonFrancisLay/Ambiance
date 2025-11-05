import io
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import librosa
import soundfile as sf  # ensures wav read support
from PIL import Image, ImageDraw
import moviepy.editor as mpy
from moviepy.audio.io.AudioFileClip import AudioFileClip
import tempfile, math

# -----------------------
# Analysis helpers
# -----------------------

@st.cache_data(show_spinner=False)
def load_audio(file_bytes: bytes) -> Tuple[np.ndarray, int]:
    # mono, native sample rate
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=None, mono=True)
    return y.astype(np.float32), int(sr)

@st.cache_data(show_spinner=False)
def compute_features(y: np.ndarray, sr: int, fps: int) -> Dict[str, np.ndarray]:
    """
    Returns a dict of time-aligned features normalized to 0..1:
    bass, mids, highs, vocals_proxy, beat_strength, and times.
    """
    hop_length = max(1, int(sr / fps))
    # STFT power
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    # bands
    bass_idx = freqs < 200
    mid_idx = (freqs >= 200) & (freqs < 2000)
    high_idx = freqs >= 2000

    # band energies over time (1-D float32)
    bass = S[bass_idx, :].mean(axis=0).astype(np.float32).ravel()
    mids = S[mid_idx, :].mean(axis=0).astype(np.float32).ravel()
    highs = S[high_idx, :].mean(axis=0).astype(np.float32).ravel()

    # vocals proxy
    # 1) harmonic component to capture pitched content
    y_harm = librosa.effects.hpss(y, kernel_size=31)[0]
    Sh = np.abs(librosa.stft(y_harm, n_fft=2048, hop_length=hop_length)) ** 2
    v_idx = (freqs >= 300) & (freqs <= 3400)
    vocals_proxy = Sh[v_idx, :].mean(axis=0).astype(np.float32).ravel()

    # beat curve (1-D)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length, units="frames")
    T = onset_env.shape[0]
    times = librosa.frames_to_time(np.arange(T), sr=sr, hop_length=hop_length)

    if beat_frames.size > 0:
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        # distance to nearest beat for each frame (shape: T,)
        dists = np.min(np.abs(times[:, None] - beat_times[None, :]), axis=1)
        beat_strength = np.exp(-5.0 * dists) * (onset_env / (onset_env.max() + 1e-9))
    else:
        beat_strength = onset_env.copy()

    beat_strength = beat_strength.astype(np.float32).ravel()
    times = times.astype(np.float32).ravel()

    # normalize to 0..1
    def norm(x):
        x = np.asarray(x, dtype=np.float32).ravel()
        mx = float(np.max(x) + 1e-9)
        return (x / mx).astype(np.float32)

    feat = {
        "times": times,
        "bass": norm(bass[:len(times)]),
        "mids": norm(mids[:len(times)]),
        "highs": norm(highs[:len(times)]),
        "vocals": norm(vocals_proxy[:len(times)]),
        "beat": norm(beat_strength[:len(times)]),
        "tempo_bpm": float(tempo),
    }
    return feat

def smooth_series(x: pd.Series, win_seconds: float, fps: int) -> pd.Series:
    # window in frames
    win_frames = max(1, int(round(win_seconds * fps)))
    return x.rolling(win_frames, min_periods=1, center=False).mean()

# -----------------------
# UI
# -----------------------

st.title("Audio feature extractor and smoother")

st.sidebar.subheader("Audio file")
audio_file = st.sidebar.file_uploader("Drop a .wav or .mp3 here or click to browse", type=["wav", "mp3", "m4a", "ogg", "flac"])

st.sidebar.subheader("Analysis settings")
fps = st.sidebar.slider("Analysis FPS (feature frames per second)", min_value=10, max_value=100, value=50, step=5)

st.sidebar.subheader("Smoothing controls")
lock_all = st.sidebar.checkbox("Lock smoothing windows", value=True)

default_win = st.sidebar.slider("All bands smoothing window (seconds)", 0.00, 20.00, 0.30, 0.05)

col1, col2 = st.columns(2)
with col1:
    bass_win = st.slider("Bass window (s)", 0.00, 2.00, default_win, 0.05, disabled=lock_all)
    mids_win = st.slider("Mids window (s)", 0.00, 2.00, default_win, 0.05, disabled=lock_all)
    highs_win = st.slider("Highs window (s)", 0.00, 2.00, default_win, 0.05, disabled=lock_all)
with col2:
    vocals_win = st.slider("Vocals window (s)", 0.00, 2.00, default_win, 0.05, disabled=lock_all)
    beat_win = st.slider("Beat window (s)", 0.00, 2.00, default_win, 0.05, disabled=lock_all)

if audio_file is None:
    st.info("Upload an audio file to begin")
    st.stop()

# Load and analyze
y, sr = load_audio(audio_file.read())
features = compute_features(y, sr, fps)

# Build DataFrame with aligned, 1-D columns
Tlen = min(
    features["times"].shape[0],
    features["bass"].shape[0],
    features["mids"].shape[0],
    features["highs"].shape[0],
    features["vocals"].shape[0],
    features["beat"].shape[0],
)
time_s = np.asarray(features["times"][:Tlen]).ravel()
bass   = np.asarray(features["bass"][:Tlen]).ravel()
mids   = np.asarray(features["mids"][:Tlen]).ravel()
highs  = np.asarray(features["highs"][:Tlen]).ravel()
vocals = np.asarray(features["vocals"][:Tlen]).ravel()
beat   = np.asarray(features["beat"][:Tlen]).ravel()

df = pd.DataFrame(
    {"time_s": time_s, "bass": bass, "mids": mids, "highs": highs, "vocals": vocals, "beat": beat}
).set_index("time_s")

# Apply smoothing
if lock_all:
    bw = mw = hw = vw = bw2 = default_win
else:
    bw, mw, hw, vw, bw2 = bass_win, mids_win, highs_win, vocals_win, beat_win

smoothed = pd.DataFrame(index=df.index)
smoothed["bass"] = smooth_series(df["bass"], bw, fps)
smoothed["mids"] = smooth_series(df["mids"], mw, fps)
smoothed["highs"] = smooth_series(df["highs"], hw, fps)
smoothed["vocals"] = smooth_series(df["vocals"], vw, fps)
smoothed["beat"] = smooth_series(df["beat"], bw2, fps)

# Show summary
st.write(f"Sample rate: {sr} Hz. Duration: {len(y) / sr:.2f} s. Estimated tempo: {features['tempo_bpm']:.1f} BPM.")

# Plot
st.subheader("Features (smoothed)")
# Streamlit line_chart expects a DataFrame; index is the x axis
st.line_chart(smoothed)

# Download smoothed CSV
csv_bytes = smoothed.reset_index().rename(columns={"time_s": "time"}).to_csv(index=False).encode("utf-8")
st.download_button(
    "Download smoothed CSV",
    data=csv_bytes,
    file_name="audio_features_smoothed.csv",
    mime="text/csv"
)

# Optional: play audio
with st.expander("Play original audio"):
    # Re-encode a short preview as wav bytes for playback if needed
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    st.audio(buf.getvalue(), format="audio/wav")

# -----------------------
# Geometric generators → video render (on demand)
# -----------------------

st.markdown("---")
st.subheader("Geometric video renderer (on demand)")

colA, colB = st.columns(2)
with colA:
    start_s = st.number_input("Start (s)", min_value=0.0, value=60.0, step=0.5, help="Start time of the segment to render.")
with colB:
    end_s = st.number_input("End (s)", min_value=0.1, value=70.0, step=0.5, help="End time of the segment to render.")

if end_s <= start_s:
    st.warning("End must be greater than Start.")
    st.stop()

st.sidebar.subheader("Render settings")
render_fps = st.sidebar.slider("Render FPS", 10, 60, 30, 1)
res_choice = st.sidebar.selectbox("Resolution", ["640x360", "854x480", "1280x720", "1920x1080"], index=1)
W, H = map(int, res_choice.split("x"))


st.sidebar.subheader("Generators (default OFF)")
layer_radial = st.sidebar.checkbox("Radial polygons & rings", value=False)
layer_spiro = st.sidebar.checkbox("Spirograph", value=False)
layer_lissajous = st.sidebar.checkbox("Lissajous", value=False)
layer_particles = st.sidebar.checkbox("Particle orbits", value=False)
layer_kaleido = st.sidebar.checkbox("Kaleidoscope overlay", value=False)
layer_flow = st.sidebar.checkbox("Flow field lines", value=False)
layer_grid = st.sidebar.checkbox("Tessellation grid", value=False)
layer_rings = st.sidebar.checkbox("Spectral rings/polygons", value=False)
layer_mandala = st.sidebar.checkbox("Mandala stamps", value=False)

 # --- Image tessellation (dissolve) ---
st.sidebar.subheader("Image tessellation (dissolve)")
layer_img_tess = st.sidebar.checkbox("Enable image tessellation", value=False)
img_files = st.sidebar.file_uploader(
    "Image(s) for tessellation (one or many)",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
    accept_multiple_files=True
)
# image selection behaviour
tess_per_tile = st.sidebar.checkbox("Random image per tile", value=True, help="If off, one image is chosen per frame for the whole grid.")
tess_img_seed = st.sidebar.number_input("Tessellation image seed", min_value=0, max_value=10**9, value=13579, step=1)

# Driving mode
tess_source = st.sidebar.selectbox("Audio source", ["beat", "bass", "mids", "highs", "vocals"], index=0)
tess_drive_mode = st.sidebar.selectbox("Drive", ["Audio only", "LFO only", "Audio × LFO", "Audio + LFO mix"], index=3)
tess_lfo_hz = st.sidebar.slider("LFO speed (Hz)", 0.00, 2.00, 0.20, 0.01)
tess_mix = st.sidebar.slider("Audio/LFO mix (Audio=1)", 0.0, 1.0, 0.70, 0.01)

# Columns evolve between min..max
tess_min_cols = st.sidebar.slider("Min columns", 1, 16, 1, 1)
tess_max_cols = st.sidebar.slider("Max columns", 1, 32, 8, 1, help="1 column is a single image; higher values create more, smaller tiles.")
tess_response = st.sidebar.slider("Response amount", 0.0, 1.0, 0.8, 0.01, help="How strongly the drive affects density.")
tess_smooth = st.sidebar.slider("Dissolve smoothness", 0.0, 1.0, 0.35, 0.01, help="Softens steps between column counts.")

# Margin evolves between min..max
tess_margin_min = st.sidebar.slider("Min tile margin (px)", 0, 40, 2, 1)
tess_margin_max = st.sidebar.slider("Max tile margin (px)", 0, 60, 12, 1)

# Pixelation
tess_pixelate = st.sidebar.checkbox("Pixelate tiles", value=True)
tess_block_min = st.sidebar.slider("Min pixel block (px)", 1, 100, 6, 1)
tess_block_max = st.sidebar.slider("Max pixel block (px)", 4, 120, 36, 1)

tess_pix_rand = st.sidebar.slider("Pixelate randomness (%)", 0, 100, 30, 1)

# --- Image fade controls ---
st.sidebar.subheader("Image fade")
img_fade_enable = st.sidebar.checkbox("Enable image fade", value=True)
img_fade_source = st.sidebar.selectbox("Fade source", ["beat", "bass", "mids", "highs", "vocals"], index=0)
img_fade_min = st.sidebar.slider("Min image opacity (%)", 0, 100, 10, 1)
img_fade_max = st.sidebar.slider("Max image opacity (%)", 0, 100, 90, 1)
img_fade_shape = st.sidebar.slider("Fade curve (0=linear, 1=sharper)", 0.0, 1.0, 0.3, 0.05)

# Prepare uploaded tessellation images (once)
tess_imgs = []
if layer_img_tess and img_files:
    for f in img_files:
        try:
            tess_imgs.append(Image.open(f).convert("RGB"))
        except Exception as e:
            st.warning(f"Could not read image: {e}")
if layer_img_tess and not tess_imgs:
    st.info("Upload at least one image to use tessellation.")

st.sidebar.subheader("Style")
trail_decay = st.sidebar.slider("Trail persistence (%)", 0, 95, 60, 1)
stroke_w = st.sidebar.slider("Base stroke width (px)", 1, 12, 2, 1)
base_opacity = st.sidebar.slider("Layer opacity (%)", 5, 100, 60, 1)


st.sidebar.subheader("Kinematics")
kin_on = st.sidebar.checkbox("Enable kinematics", value=True)
source_opts = ["none", "bass", "mids", "highs", "vocals", "beat"]
kin_pos_src_x = st.sidebar.selectbox("Position X source", source_opts, index=1)
kin_pos_src_y = st.sidebar.selectbox("Position Y source", source_opts, index=2)
kin_size_src = st.sidebar.selectbox("Size source", source_opts, index=3)
kin_rot_src = st.sidebar.selectbox("Rotation source", source_opts, index=5)
kin_opacity_src = st.sidebar.selectbox("Opacity source", source_opts, index=0)
kin_speed_hz = st.sidebar.slider("Motion speed (Hz)", 0.0, 1.0, 0.5, 0.01)
kin_pos_amp_pct = st.sidebar.slider("Position amplitude (% of min dimension)", 0.0, 100.0, 20.0, 1.0)
kin_size_range_pct = st.sidebar.slider("Size range (±%)", 0.0, 200.0, 30.0, 1.0)
kin_rot_max_deg = st.sidebar.slider("Rotation max (deg)", 0.0, 180.0, 30.0, 1.0)
kin_opacity_pct = st.sidebar.slider("Opacity modulation (±%)", 0.0, 100.0, 30.0, 1.0)

# --- Population fill/empty ---
st.sidebar.subheader("Population fill/empty")
pop_enable = st.sidebar.checkbox("Enable population (stamped objects)", value=True)
pop_source = st.sidebar.selectbox("Population source", ["vocals", "bass", "mids", "highs", "beat"], index=0, help="Controls fill (spawn) rate; low source empties the screen as objects expire.")
pop_spawn_rate = st.sidebar.slider("Spawn rate @ source=1 (objects/sec)", 0.0, 20.0, 4.0, 0.1)
pop_max_objs = st.sidebar.slider("Max population (objects)", 10, 2000, 400, 10)
pop_life_sec = st.sidebar.slider("Mean lifetime (s)", 0.5, 20.0, 6.0, 0.1)
pop_life_jitter = st.sidebar.slider("Lifetime jitter (±%)", 0.0, 100.0, 30.0, 1.0)
pop_size_min = st.sidebar.slider("Min size (px)", 2, 200, 12, 1)
pop_size_max = st.sidebar.slider("Max size (px)", 4, 400, 60, 1)
pop_shapes = st.sidebar.multiselect("Stamp shapes", ["circle", "ring", "triangle", "poly"], default=["circle", "ring", "triangle"])
pop_seed = st.sidebar.number_input("Population seed", min_value=0, max_value=10**9, value=777, step=1)

# --- Per-layer kinematic variations ---
st.sidebar.subheader("Per-layer variation")
rand_variations = st.sidebar.checkbox("Randomize per-layer kinematic variations", value=True)
var_seed = st.sidebar.number_input("Variation seed", min_value=0, max_value=10**9, value=2025, step=1)
reroll_vars = st.sidebar.button("Re-roll variations")
# --- Population system helpers ---

def _rand_color_from_palette(rng):
    return COLORS[int(rng.integers(0, len(COLORS)))]

def _draw_stamp(draw: ImageDraw.ImageDraw, shape: str, cx: float, cy: float, size: float, color: tuple, alpha: int, rotation_rad: float = 0.0, stroke_w: int = 1):
    col = color + (int(np.clip(alpha, 0, 255)),)
    if shape == "circle":
        r = size * 0.5
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=stroke_w)
    elif shape == "ring":
        r = size * 0.5
        # outer and inner
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=max(1, stroke_w))
        r2 = max(0, r - max(1, stroke_w*2))
        if r2 > 0:
            draw.ellipse([cx - r2, cy - r2, cx + r2, cy + r2], outline=col, width=1)
    elif shape == "triangle":
        r = size * 0.6
        ang0 = rotation_rad
        pts = []
        for k in range(3):
            ang = ang0 + 2 * math.pi * k / 3.0
            pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
        draw.polygon(pts, outline=col, width=stroke_w)
    else:  # "poly"
        r = size * 0.6
        sides = 5 + int(size) % 4  # 5..8
        ang0 = rotation_rad
        pts = []
        for k in range(sides):
            ang = ang0 + 2 * math.pi * k / sides
            pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
        draw.polygon(pts, outline=col, width=stroke_w)

def _update_population(pop_state: dict, t_abs: float, F: dict, W: int, H: int, params: dict):
    """
    pop_state: { 'objs': list, 'last_t': float, 'rng': Generator }
    Each obj: { 'birth': float, 'life': float, 'x': float, 'y': float, 'size': float, 'shape': str, 'color': (r,g,b), 'rot': float }
    params: dict with keys used below.
    """
    if not params.get("enable", False):
            return
    objs = pop_state["objs"]
    rng = pop_state["rng"]
    last_t = pop_state["last_t"]
    if last_t is None:
        pop_state["last_t"] = t_abs
        return
    dt = max(0.0, t_abs - last_t)
    pop_state["last_t"] = t_abs

    # remove dead
    alive = []
    for o in objs:
        age = t_abs - o["birth"]
        if age <= o["life"]:
            alive.append(o)
    objs[:] = alive

    # spawn
    src_val = float(F.get(params["source"], 0.0))
    rate = float(params["spawn_rate"]) * np.clip(src_val, 0.0, 1.0)
    expect_new = rate * dt
    n_new = int(expect_new)
    if rng.random() < (expect_new - n_new):
        n_new += 1

    # bound population
    space = max(0, int(params["max_objs"]) - len(objs))
    n_new = min(n_new, space)
    if n_new <= 0:
        return

    shapes = params["shapes"] if params["shapes"] else ["circle"]
    jitter = float(params["life_jitter"])
    life_mean = float(params["life_sec"])

    for _ in range(n_new):
        life = life_mean * (1.0 + (jitter / 100.0) * (rng.random() * 2 - 1))
        life = max(0.2, life)
        x = float(rng.integers(0, W))
        y = float(rng.integers(0, H))
        size = float(rng.uniform(params["size_min"], params["size_max"]))
        shape = shapes[int(rng.integers(0, len(shapes)))]
        color = _rand_color_from_palette(rng)
        rot = float(rng.uniform(0, 2 * math.pi))
        objs.append({"birth": t_abs, "life": life, "x": x, "y": y, "size": size, "shape": shape, "color": color, "rot": rot})

def _draw_population(draw: ImageDraw.ImageDraw, pop_state: dict, t_abs: float, opacity_scale: float = 1.0, stroke_w: int = 1):
    objs = pop_state["objs"]
    for o in objs:
        age = max(0.0, t_abs - o["birth"])
        a = 1.0 - min(1.0, age / max(1e-6, o["life"]))  # linear fade out
        alpha = int(255 * np.clip(a * opacity_scale, 0.0, 1.0))
        _draw_stamp(draw, o["shape"], o["x"], o["y"], o["size"], o["color"], alpha, o["rot"], stroke_w=stroke_w)

LAYER_KEYS = [
    ("grid", "Tessellation grid"),
    ("radial", "Radial polygons & rings"),
    ("rings", "Spectral rings/polygons"),
    ("lissajous", "Lissajous"),
    ("spiro", "Spirograph"),
    ("particles", "Particle orbits"),
    ("flow", "Flow field lines"),
    ("mandala", "Mandala stamps"),
    ("kaleido", "Kaleidoscope overlay"),
]

def _build_layer_variations(seed: int, use_random: bool) -> dict:
    rng = np.random.default_rng(seed)
    # candidate sources to override from; exclude "none"
    sources = ["bass", "mids", "highs", "vocals", "beat"]
    vars_dict = {}
    for i, (key, _label) in enumerate(LAYER_KEYS):
        cfg = {
            # axis behaviour
            "swap_xy": False,
            "invert_x": False,
            "invert_y": False,
            # parameter inversions
            "invert_size": False,
            "invert_rot": False,
            "invert_opacity": False,
            # optional source overrides (None = use global selection)
            "src_x": None,
            "src_y": None,
            "src_size": None,
            "src_rot": None,
            "src_opacity": None,
        }
        if use_random:
            # alternate signs across layers to create contrast
            cfg["invert_x"] = bool(rng.integers(0, 2))
            cfg["invert_y"] = bool(rng.integers(0, 2))
            cfg["swap_xy"] = bool(rng.integers(0, 2))
            cfg["invert_size"] = bool(rng.integers(0, 2))
            cfg["invert_rot"] = bool(rng.integers(0, 2))
            cfg["invert_opacity"] = bool(rng.integers(0, 2))
            # with some probability, remap sources
            if rng.random() < 0.35:
                cfg["src_x"] = sources[int(rng.integers(0, len(sources)))]
            if rng.random() < 0.35:
                cfg["src_y"] = sources[int(rng.integers(0, len(sources)))]
            if rng.random() < 0.25:
                cfg["src_size"] = sources[int(rng.integers(0, len(sources)))]
            if rng.random() < 0.25:
                cfg["src_rot"] = sources[int(rng.integers(0, len(sources)))]
            if rng.random() < 0.25:
                cfg["src_opacity"] = sources[int(rng.integers(0, len(sources)))]
        vars_dict[key] = cfg
    return vars_dict

# Initialize / reroll session state
if "layer_vars" not in st.session_state:
    st.session_state["layer_vars"] = _build_layer_variations(var_seed, rand_variations)
elif reroll_vars:
    st.session_state["layer_vars"] = _build_layer_variations(var_seed, rand_variations)
elif st.session_state.get("layer_vars_seed") != var_seed or st.session_state.get("layer_vars_flag") != rand_variations:
    st.session_state["layer_vars"] = _build_layer_variations(var_seed, rand_variations)

st.session_state["layer_vars_seed"] = var_seed
st.session_state["layer_vars_flag"] = rand_variations

layer_vars = st.session_state["layer_vars"]

# Build fast feature interpolators from smoothed curves
idx_times = smoothed.index.values.astype(np.float32)
vals = {k: smoothed[k].values.astype(np.float32) for k in ["bass", "mids", "highs", "vocals", "beat"]}

def feat_at(t: float):
    t_clamped = float(np.clip(t, idx_times[0], idx_times[-1]))
    out = {}
    for k, v in vals.items():
        out[k] = float(np.interp(t_clamped, idx_times, v))
    return out


# Simple palette helper
COLORS = [(255, 200, 40), (100, 200, 255), (255, 80, 140), (120, 255, 120), (240, 240, 240)]

def _apply_alpha(img_rgba: Image.Image, scale: float) -> Image.Image:
    """Multiply existing alpha by scale in 0..1 and return a new RGBA image."""
    scale = float(np.clip(scale, 0.0, 1.0))
    if scale >= 0.999:
        return img_rgba
    if scale <= 0.0:
        return Image.new("RGBA", img_rgba.size, (0, 0, 0, 0))
    r, g, b, a = img_rgba.split()
    a_np = np.array(a, dtype=np.float32)
    a_np = np.clip(a_np * scale, 0, 255).astype(np.uint8)
    a2 = Image.fromarray(a_np, mode="L")
    return Image.merge("RGBA", (r, g, b, a2))

# --- Tessellation helpers ---
def _pixelate_np(img_np: np.ndarray, block: int) -> np.ndarray:
    if block <= 1:
        return img_np
    h, w, c = img_np.shape
    # ensure divisible
    bw = max(1, int(block))
    nh = max(1, h // bw)
    nw = max(1, w // bw)
    small = Image.fromarray(img_np).resize((nw, nh), Image.NEAREST)
    return np.array(small.resize((w, h), Image.NEAREST))

def _blend_images_rgba(base: Image.Image, overlay: Image.Image, alpha: float):
    # alpha 0..1
    if alpha <= 0:
        return base
    if alpha >= 1:
        base.alpha_composite(overlay)
        return base
    ov = overlay.copy()
    a = Image.new("L", ov.size, int(255 * np.clip(alpha, 0.0, 1.0)))
    r, g, b, _ = ov.split()
    ov = Image.merge("RGBA", (r, g, b, a))
    base.alpha_composite(ov)
    return base

def _tessellate_image_layer(src_imgs: list, W: int, H: int, cols_float: float, margin_px: int,
                            pixelate_on: bool, block_px: int,
                            per_tile: bool, rng: np.random.Generator) -> Image.Image:
    """
    Build a grid of the source images. cols_float can be fractional; we render floor and ceil and crossfade.
    Allows multiple images and random selection.
    """
    cols_float = float(np.clip(cols_float, 1.0, max(1.0, cols_float)))
    c0 = int(np.floor(cols_float))
    c1 = max(1, min(int(np.ceil(cols_float)), 64))
    frac = cols_float - c0

    def make_grid(cols: int) -> Image.Image:
        if cols <= 0:
            cols = 1
        rows = max(1, int(round(cols * H / max(1, W))))  # preserve aspect roughly square tiles
        # compute tile size
        tile_w = max(1, int((W - (cols + 1) * margin_px) / max(1, cols)))
        tile_h = max(1, int((H - (rows + 1) * margin_px) / max(1, rows)))

        grid = Image.new("RGBA", (W, H), (0, 0, 0, 0))

        if not src_imgs:
            return grid
        # choose a single image for the whole grid (per frame) if per_tile is False
        if per_tile:
            chosen_for_frame = None
        else:
            chosen_for_frame = src_imgs[int(rng.integers(0, len(src_imgs)))]
            chosen_for_frame = chosen_for_frame.resize((tile_w, tile_h), Image.LANCZOS)

        for r in range(rows):
            for c in range(cols):
                x = margin_px + c * (tile_w + margin_px)
                y = margin_px + r * (tile_h + margin_px)
                if per_tile:
                    base_img = src_imgs[int(rng.integers(0, len(src_imgs)))]
                    img_fit = base_img.resize((tile_w, tile_h), Image.LANCZOS)
                else:
                    img_fit = chosen_for_frame
                tile = img_fit
                if pixelate_on and block_px > 1:
                    np_tile = np.array(tile)
                    np_tile = _pixelate_np(np_tile, int(block_px))
                    tile = Image.fromarray(np_tile)
                grid.alpha_composite(tile.convert("RGBA"), (x, y))
        return grid

    g0 = make_grid(c0)
    if c1 == c0:
        return g0
    g1 = make_grid(c1)
    # crossfade between c0 and c1 to create dissolve as cols_float changes
    out = g0.copy()
    return _blend_images_rgba(out, g1, frac)

def _src(F: dict, name: str) -> float:
    if not name or name == "none":
        return 0.0
    return float(F.get(name, 0.0))

def draw_layers(draw: ImageDraw.ImageDraw, t: float, F: dict, W: int, H: int, seed: int = 123,
                kin_on: bool = False,
                kin_speed_hz: float = 0.5,
                kin_pos_src_x: str = "none",
                kin_pos_src_y: str = "none",
                kin_size_src: str = "none",
                kin_rot_src: str = "none",
                kin_opacity_src: str = "none",
                kin_pos_amp_pct: float = 20.0,
                kin_size_range_pct: float = 30.0,
                kin_rot_max_deg: float = 30.0,
                kin_opacity_pct: float = 30.0):
    rng = np.random.default_rng(seed + int(t * 1000))
    base_cx, base_cy = W / 2, H / 2
    base_r = 0.3 * min(W, H)
    color = COLORS[int((F["bass"] + 2*F["mids"] + 3*F["highs"]) * 3) % len(COLORS)]
    a_base = max(8, int(255 * (base_opacity / 100.0)))

    # capture per-layer variations from outer scope
    global layer_vars  # uses st.session_state['layer_vars'] via closure in render_frame

    def kin_params(cfg: dict):
        cx, cy, r_local = base_cx, base_cy, base_r
        a_mult = 1.0
        krot = 0.0
        if not kin_on:
            return cx, cy, r_local, a_base, krot
        # choose sources (override or global)
        src_x = cfg.get("src_x") or kin_pos_src_x
        src_y = cfg.get("src_y") or kin_pos_src_y
        src_size = cfg.get("src_size") or kin_size_src
        src_rot = cfg.get("src_rot") or kin_rot_src
        src_op = cfg.get("src_opacity") or kin_opacity_src

        phase = 2.0 * math.pi * kin_speed_hz * t
        amp_px = (kin_pos_amp_pct / 100.0) * float(min(W, H)) * 0.5

        sx = 2.0 * _src(F, src_x) - 1.0
        sy = 2.0 * _src(F, src_y) - 1.0
        # axis swap and inversion
        if cfg.get("swap_xy", False):
            sx, sy = sy, sx
        if cfg.get("invert_x", False):
            sx = -sx
        if cfg.get("invert_y", False):
            sy = -sy

        kx = amp_px * sx * math.sin(phase)
        ky = amp_px * sy * math.cos(phase + math.pi * 0.25)
        cx += kx
        cy += ky

        # size
        sraw = (2.0 * _src(F, src_size) - 1.0)
        if cfg.get("invert_size", False):
            sraw = -sraw
        kscale = 1.0 + (kin_size_range_pct / 100.0) * sraw
        r_local *= kscale

        # rotation
        rraw = (2.0 * _src(F, src_rot) - 1.0)
        if cfg.get("invert_rot", False):
            rraw = -rraw
        krot = math.radians(kin_rot_max_deg) * rraw * math.sin(phase)

        # opacity
        oraw = (2.0 * _src(F, src_op) - 1.0)
        if cfg.get("invert_opacity", False):
            oraw = -oraw
        a_local = int(np.clip(a_base * (1.0 + (kin_opacity_pct / 100.0) * oraw), 8, 255))
        return cx, cy, r_local, a_local, krot

    if layer_grid:
        cx, cy, r_base_l, a, krot = kin_params(layer_vars.get("grid", {}))
        cell = int(20 + 80 * F["highs"])
        for x in range(0, W, cell):
            draw.line([(x, 0), (x, H)], fill=color + (a,), width=1)
        for y in range(0, H, cell):
            draw.line([(0, y), (W, y)], fill=color + (a,), width=1)

    if layer_radial:
        cx, cy, r_base_l, a, krot = kin_params(layer_vars.get("radial", {}))
        spokes = max(3, int(4 + 20 * F["highs"]))
        for i in range(3):
            r = r_base_l * (0.4 + 0.25 * i) * (1.0 + 0.5 * F["bass"])
            ang0 = 2 * math.pi * (i * 0.07 + 0.1 * F["mids"] * t) + krot
            pts = [(cx + r * math.cos(ang0 + 2*math.pi*k/spokes), cy + r * math.sin(ang0 + 2*math.pi*k/spokes)) for k in range(spokes)]
            draw.polygon(pts, outline=color + (a,), width=stroke_w)

    if layer_rings:
        cx, cy, r_base_l, a, krot = kin_params(layer_vars.get("rings", {}))
        for s, w in [(1.0 + 0.8*F["bass"], 4), (0.7 + 0.6*F["mids"], 2)]:
            r = r_base_l * s
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color + (a,), width=max(1, stroke_w - (w//4)))

    if layer_lissajous:
        cx, cy, r_base_l, a, krot = kin_params(layer_vars.get("lissajous", {}))
        Ax = r_base_l * (0.5 + F["bass"]) ; Ay = r_base_l * (0.5 + F["mids"])
        fx = 2 + int(5 * F["highs"]) ; fy = 3 + int(4 * F["beat"])
        ph = 2 * math.pi * (0.1 * t)
        pts = []
        cosr, sinr = math.cos(krot), math.sin(krot)
        for k in range(400):
            tt = k / 400.0 * 2 * math.pi
            x0 = Ax * math.sin(fx * tt + ph)
            y0 = Ay * math.sin(fy * tt)
            x = cx + x0 * cosr - y0 * sinr
            y = cy + x0 * sinr + y0 * cosr
            pts.append((x, y))
        draw.line(pts, fill=color + (a,), width=stroke_w)

    if layer_spiro:
        cx, cy, r_base_l, a, krot = kin_params(layer_vars.get("spiro", {}))
        R = r_base_l * (0.4 + 0.4 * F["bass"])
        r = R * (0.3 + 0.3 * F["mids"])
        p = r * (0.3 + 0.5 * F["highs"])
        pts = []
        cosr, sinr = math.cos(krot), math.sin(krot)
        for k in range(800):
            t2 = k / 800.0 * 2 * math.pi
            x0 = (R - r) * math.cos(t2) + p * math.cos(((R - r) / r) * t2)
            y0 = (R - r) * math.sin(t2) - p * math.sin(((R - r) / r) * t2)
            x = cx + x0 * cosr - y0 * sinr
            y = cy + x0 * sinr + y0 * cosr
            pts.append((x, y))
        draw.line(pts, fill=color + (a,), width=stroke_w)

    if layer_particles:
        cx, cy, r_base_l, a, krot = kin_params(layer_vars.get("particles", {}))
        n = 50
        for i in range(n):
            ang = 2 * math.pi * (i / n + 0.05 * F["mids"] * t) + krot
            rad = r_base_l * (0.2 + 0.6 * F["bass"]) + 20 * math.sin(ang * (2 + 4 * F["highs"]))
            x = cx + rad * math.cos(ang)
            y = cy + rad * math.sin(ang)
            rdot = 2 + 3 * F["highs"]
            draw.ellipse([x - rdot, y - rdot, x + rdot, y + rdot], outline=color + (a,), width=1)

    if layer_flow:
        cx, cy, r_base_l, a, krot = kin_params(layer_vars.get("flow", {}))
        step = int(30 + 60 * (1 - F["mids"]))
        L = int(10 + 30 * F["highs"])
        for y0 in range(step // 2, H, step):
            for x0 in range(step // 2, W, step):
                ang = 6.28318 * (math.sin(0.001 * x0 + 0.7 * F["mids"]) + math.cos(0.001 * y0 + 0.9 * F["bass"])) + krot
                x1 = x0 + L * math.cos(ang)
                y1 = y0 + L * math.sin(ang)
                draw.line([(x0, y0), (x1, y1)], fill=color + (a,), width=1)

    if layer_mandala:
        cx, cy, r_base_l, a, krot = kin_params(layer_vars.get("mandala", {}))
        spokes = max(6, int(8 + 20 * F["highs"]))
        rings = 3
        for i in range(rings):
            r = r_base_l * (0.2 + 0.25 * i) * (1 + 0.5 * F["bass"])
            for k in range(spokes):
                ang = 2 * math.pi * k / spokes + 0.5 * F["mids"] * t + krot
                x = cx + r * math.cos(ang)
                y = cy + r * math.sin(ang)
                draw.line([(cx, cy), (x, y)], fill=color + (a,), width=stroke_w)

    if layer_kaleido:
        # simple 4-way mirror for now (lightweight)
        # Optionally: cx, cy, r_base_l, a, krot = kin_params(layer_vars.get("kaleido", {}))
        pass  # placeholder; heavy kaleidoscope can be added later


def render_frame(t_abs: float, bg_img: Image.Image | None, trail: Image.Image | None, pop_state: dict | None, pop_params: dict | None):
    F = feat_at(t_abs)
    # background: solid black
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    draw = ImageDraw.Draw(canvas, "RGBA")

    # optional image tessellation background
    if layer_img_tess and tess_imgs:
        # audio drive
        a_val = float(F.get(tess_source, 0.0))
        # lfo drive
        lfo = 0.5 * (1.0 + math.sin(2.0 * math.pi * float(tess_lfo_hz) * t_abs))
        # combine per mode
        if tess_drive_mode == "Audio only":
            drive = a_val
        elif tess_drive_mode == "LFO only":
            drive = lfo
        elif tess_drive_mode == "Audio × LFO":
            drive = a_val * lfo
        else:  # "Audio + LFO mix"
            drive = float(tess_mix) * a_val + (1.0 - float(tess_mix)) * lfo

        # easing and response
        drive = float(np.clip(drive, 0.0, 1.0))
        eased = drive ** (1.0 - tess_smooth)
        # columns in [min..max]
        cmin = min(int(tess_min_cols), int(tess_max_cols))
        cmax = max(int(tess_min_cols), int(tess_max_cols))
        cols_target = float(cmin) + (float(cmax - cmin)) * (eased * tess_response)

        # margin in [min..max]
        mmin = min(int(tess_margin_min), int(tess_margin_max))
        mmax = max(int(tess_margin_min), int(tess_margin_max))
        margin_now = int(round(mmin + (mmax - mmin) * drive))

        # pixel block in [min..max] with random jitter
        bmin = max(1, int(tess_block_min))
        bmax = max(bmin, int(tess_block_max))
        base_block = int(round(bmin + (bmax - bmin) * drive))
        jitter = (tess_pix_rand / 100.0) * (np.random.uniform(-1.0, 1.0))
        block_now = int(np.clip(base_block * (1.0 + jitter), 1, 512))

        rng_loc = np.random.default_rng(int(tess_img_seed + t_abs * 10_000))
        grid_img = _tessellate_image_layer(
            tess_imgs, W, H, cols_target, margin_now,
            tess_pixelate, block_now,
            tess_per_tile, rng_loc
        )
        if img_fade_enable:
            fsrc = float(F.get(img_fade_source, 0.0))
            # apply shaping: raise to power for sharper response when img_fade_shape is higher
            shaped = fsrc ** (1.0 + img_fade_shape * 3.0)
            a_min = img_fade_min / 100.0
            a_max = max(a_min, img_fade_max / 100.0)
            alpha_scale = a_min + (a_max - a_min) * shaped
            grid_img = _apply_alpha(grid_img, alpha_scale)
        canvas.alpha_composite(grid_img)

    # draw shapes
    draw_layers(
        draw, t_abs, F, W, H,
        kin_on=kin_on,
        kin_speed_hz=kin_speed_hz,
        kin_pos_src_x=kin_pos_src_x,
        kin_pos_src_y=kin_pos_src_y,
        kin_size_src=kin_size_src,
        kin_rot_src=kin_rot_src,
        kin_opacity_src=kin_opacity_src,
        kin_pos_amp_pct=kin_pos_amp_pct,
        kin_size_range_pct=kin_size_range_pct,
        kin_rot_max_deg=kin_rot_max_deg,
        kin_opacity_pct=kin_opacity_pct,
    )

    # population: update and draw
    if pop_state is not None and pop_params is not None and pop_params.get("enable", False):
        _update_population(pop_state, t_abs, F, W, H, pop_params)
        _draw_population(draw, pop_state, t_abs, opacity_scale=base_opacity / 100.0, stroke_w=max(1, stroke_w))

    # trails: simple exponential decay
    if trail is not None:
        decay = max(0.0, min(0.95, trail_decay / 100.0))
        trail = trail.convert("RGBA")
        # blend old trail with new canvas
        tr = trail.copy()
        tr.putalpha(int(255 * decay))
        canvas.alpha_composite(tr)
    return canvas

process = st.button("Process", type="primary")

if process:
    # clamp bounds
    start_s = max(0.0, float(start_s))
    end_s = float(end_s)
    duration = max(0.1, end_s - start_s)

    progress = st.progress(0, text="Rendering frames...")

    # Prepare audio segment via temp WAV to preserve SR
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as aw:
        s0 = int(start_s * sr)
        s1 = int(end_s * sr)
        yseg = y[s0:s1]
        if yseg.size == 0:
            st.error("Empty audio segment; check start/end times.")
            st.stop()
        sf.write(aw.name, yseg.astype(np.float32), sr)
        audio_clip = AudioFileClip(aw.name)

    # Stateful trail buffer inside make_frame closure
    trail_holder = {"img": None}
    total_frames = int(duration * render_fps)

    # Population state (persists across frames)
    pop_state = {"objs": [], "last_t": None, "rng": np.random.default_rng(int(pop_seed))}
    pop_params = {
        "enable": bool(pop_enable),
        "source": str(pop_source),
        "spawn_rate": float(pop_spawn_rate),
        "max_objs": int(pop_max_objs),
        "life_sec": float(pop_life_sec),
        "life_jitter": float(pop_life_jitter),
        "size_min": int(pop_size_min),
        "size_max": int(pop_size_max),
        "shapes": list(pop_shapes),
    }

    def make_frame_local(t_local: float):
        # t_local in [0, duration)
        t_abs = start_s + float(t_local)
        frame_img = render_frame(t_abs, None, trail_holder["img"], pop_state, pop_params)  # draw
        # update trail to current canvas
        trail_holder["img"] = frame_img.copy()
        # update progress
        fidx = min(total_frames - 1, int(t_local * render_fps))
        progress.progress(min(1.0, (fidx + 1) / max(1, total_frames)), text=f"Rendering frame {fidx+1}/{total_frames}")
        return np.array(frame_img.convert("RGB"))

    clip = mpy.VideoClip(make_frame_local, duration=duration)
    clip = clip.set_audio(audio_clip)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpv:
        clip.write_videofile(tmpv.name, fps=render_fps, codec="libx264", audio_codec="aac", verbose=False, logger=None)
        data = open(tmpv.name, "rb").read()

    st.success("Done.")
    st.video(data)
    st.download_button("Download video", data=data, file_name="geomart.mp4", mime="video/mp4")