import streamlit as st
import numpy as np
import tempfile
import librosa
import moviepy.editor as mpy
from PIL import Image, ImageDraw
from io import BytesIO
import base64
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
import soundfile as sf

# -----------------------
# Helpers
# -----------------------

@st.cache_data
def load_audio(file_bytes):
    # Load audio into mono float32 and sample rate using librosa
    y, sr = librosa.load(BytesIO(file_bytes), sr=None, mono=True)
    return y, sr

@st.cache_data
def analyze_audio(y, sr, fps):
    # Break audio into frames aligned with video frames
    hop_length = int(sr / fps)  # samples per video frame
    frame_starts = np.arange(0, len(y), hop_length)

    amp = []
    bass = []
    mids = []
    highs = []

    # Precompute STFT for spectral energy
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))**2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    # frequency band indices
    bass_idx = freqs < 200
    mid_idx  = (freqs >= 200) & (freqs < 2000)
    hi_idx   = freqs >= 2000

    # Global tempo estimate and beat envelope
    # tempo: BPM, beats: frame indices (in librosa time base)
    tempo_bpm, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    # convert beat frame positions to time (s)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    # build a simple "beat strength" envelope aligned with our frame_starts:
    # for each video frame time, find distance to nearest beat in seconds.
    frame_times_sec = frame_starts / sr
    beat_strength = []
    for tsec in frame_times_sec:
        if len(beat_times) == 0:
            beat_strength.append(0.0)
        else:
            dist = np.min(np.abs(beat_times - tsec))
            # closer to beat -> stronger, exponential falloff
            beat_strength.append(float(np.exp(-4.0 * dist)))
    beat_strength = np.array(beat_strength)

    # match each time frame in S with our frame list
    for t_i in range(min(len(frame_starts), S.shape[1])):
        frame_samples = y[frame_starts[t_i]:frame_starts[t_i]+hop_length]
        if len(frame_samples) == 0:
            break

        amp.append(float(np.sqrt(np.mean(frame_samples**2)) + 1e-9))

        spec_col = S[:, t_i]
        bass.append(float(np.mean(spec_col[bass_idx]) + 1e-9))
        mids.append(float(np.mean(spec_col[mid_idx]) + 1e-9))
        highs.append(float(np.mean(spec_col[hi_idx]) + 1e-9))

    # normalise 0..1
    def norm(v):
        v = np.array(v)
        v = v / (np.max(v) + 1e-9)
        return v

    features = {
        "amp": norm(amp),
        "bass": norm(bass),
        "mids": norm(mids),
        "highs": norm(highs),
        "times": frame_starts[:len(amp)] / sr,
        # tempo info
        "tempo_bpm": float(tempo_bpm),
        # beat indices and proximity
        "beat_times": beat_times,
        "beat_strength": beat_strength[:len(amp)]
    }
    return features

# --- Tempo-aware hold/crossfade helper ---
def tempo_params(features, idx):
    """
    Given the global tempo (BPM) and local beat strength,
    return:
      hold_factor   : how long to 'freeze' a frame,
      blend_softness: how gentle the fade between held frames should be.
    Slower tempo -> longer holds, softer blend.
    Faster tempo -> more flicker, snappier cuts.
    """
    tempo_bpm = features.get("tempo_bpm", 120.0)
    beat_local = features.get("beat_strength", [0.0])[idx]

    # Map tempo to a 0..1 "slowness": 60 BPM => 1.0, 180 BPM => 0.0 (clipped)
    slowness = np.clip((180.0 - tempo_bpm) / 120.0, 0.0, 1.0)

    # base hold in seconds: between ~0.03s and ~0.3s depending on slowness
    base_hold = 0.03 + 0.27 * slowness

    # if we're near a beat (beat_local high), shorten hold to let it update
    hold_factor = base_hold * (1.0 - 0.5 * beat_local)

    # blend softness 0..1 : 0 = hard cut, 1 = very soft fade
    blend_softness = 0.2 + 0.6 * slowness
    # if on beat, reduce softness slightly to feel snappier
    blend_softness *= (1.0 - 0.3 * beat_local)

    return hold_factor, blend_softness

def select_image_pair(t,
                      repeat_period,
                      imgs,
                      randomize_images=True):
    """
    Decide which two images (A,B) to use for this time t.

    We divide time into periods of length repeat_period.
    For each period, pick a pair of images:
      - if randomize_images: pseudo-random but stable for that period
      - else: walk through imgs in order

    Returns (imgA_np, imgB_np) as numpy arrays, same size.
    """

    if len(imgs) == 1:
        # Only one image, use it for both
        imgA_np = imgs[0]
        imgB_np = imgs[0]
        return imgA_np, imgB_np

    period_index = int(t // repeat_period)

    if randomize_images:
        # stable random per period: seed by period index
        rng = np.random.default_rng(seed=period_index)
        idxs = rng.choice(len(imgs), size=2, replace=False)
        idxA, idxB = int(idxs[0]), int(idxs[1])
    else:
        # sequential pair: (0->1), (1->2), ... wrap around
        idxA = period_index % len(imgs)
        idxB = (period_index + 1) % len(imgs)

    imgA_np = imgs[idxA]
    imgB_np = imgs[idxB]

    # force same size (resize B to A's size)
    if imgB_np.shape[:2] != imgA_np.shape[:2]:
        pilB = Image.fromarray(imgB_np)
        pilB = pilB.resize((imgA_np.shape[1], imgA_np.shape[0]))
        imgB_np = np.array(pilB)

    return imgA_np, imgB_np


# --- Extra geometric and color helpers ---
def kaleidoscope(frame, wedges=6, rot=0.0):
    """
    Mirror wedges around center to create symmetric tiling.
    wedges: integer >= 1
    rot: radians, rotation of the kaleidoscope
    """
    H, W, _ = frame.shape
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W * 0.5, H * 0.5
    x = xx - cx
    y = yy - cy
    a = np.arctan2(y, x) + rot
    # tile angle into one wedge
    wedge_angle = (2.0 * np.pi) / max(1, int(wedges))
    a_tile = np.mod(a, wedge_angle)
    r = np.sqrt(x * x + y * y)
    X = cx + r * np.cos(a_tile)
    Y = cy + r * np.sin(a_tile)
    X = np.clip(X, 0, W - 1).astype(np.int32)
    Y = np.clip(Y, 0, H - 1).astype(np.int32)
    return frame[Y, X]

def polar_warp(frame, r_amt=0.1, a_amt=0.1):
    """
    Simple polar-space warp: push-pull radius and swirl angle.
    r_amt: radial distortion amount
    a_amt: angular distortion amount
    """
    H, W, _ = frame.shape
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W * 0.5, H * 0.5
    # normalize to -1..1 in the larger dimension
    scale = float(max(W, H))
    xn = (xx - cx) / scale
    yn = (yy - cy) / scale
    r = np.sqrt(xn * xn + yn * yn)
    a = np.arctan2(yn, xn)
    # ease radial so center moves more than edges
    r2 = np.clip(r + r_amt * (1.0 - r * r), 0.0, 1.0)
    a2 = a + a_amt * np.sin(6.0 * a + 4.0 * r)
    X = cx + (r2 * np.cos(a2)) * scale
    Y = cy + (r2 * np.sin(a2)) * scale
    X = np.clip(X, 0, W - 1).astype(np.int32)
    Y = np.clip(Y, 0, H - 1).astype(np.int32)
    return frame[Y, X]

def _rgb_to_hsv_np(rgb):
    """
    Vectorized RGB (0..255 uint8 or float) to HSV where:
      H in degrees 0..360, S in 0..1, V in 0..1
    """
    f = rgb.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    cmax = np.max(f, axis=-1)
    cmin = np.min(f, axis=-1)
    delta = cmax - cmin + 1e-12

    h = np.zeros_like(cmax)
    mask = delta > 0
    r_eq = (cmax == r) & mask
    g_eq = (cmax == g) & mask
    b_eq = (cmax == b) & mask

    h[r_eq] = (60.0 * ((g[r_eq] - b[r_eq]) / delta[r_eq]) + 360.0) % 360.0
    h[g_eq] = (60.0 * ((b[g_eq] - r[g_eq]) / delta[g_eq]) + 120.0) % 360.0
    h[b_eq] = (60.0 * ((r[b_eq] - g[b_eq]) / delta[b_eq]) + 240.0) % 360.0

    s = np.where(cmax <= 0.0, 0.0, delta / (cmax + 1e-12))
    v = cmax
    return h, s, v

def _hsv_to_rgb_np(h, s, v):
    """
    HSV to RGB inverse of _rgb_to_hsv_np. H in degrees.
    """
    h = (h % 360.0) / 60.0  # 0..6
    c = v * s
    x = c * (1.0 - np.abs(np.mod(h, 2.0) - 1.0))
    z = np.zeros_like(h)

    # choose sector
    conditions = [
        (0 <= h) & (h < 1),
        (1 <= h) & (h < 2),
        (2 <= h) & (h < 3),
        (3 <= h) & (h < 4),
        (4 <= h) & (h < 5),
        (5 <= h) & (h <= 6),
    ]
    rgb_primes = [
        (c, x, z),
        (x, c, z),
        (z, c, x),
        (z, x, c),
        (x, z, c),
        (c, z, x),
    ]

    rp = np.zeros_like(h)
    gp = np.zeros_like(h)
    bp = np.zeros_like(h)
    for cond, (rpi, gpi, bpi) in zip(conditions, rgb_primes):
        rp = np.where(cond, rpi, rp)
        gp = np.where(cond, gpi, gp)
        bp = np.where(cond, bpi, bp)

    m = v - c
    r = rp + m
    g = gp + m
    b = bp + m
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

def hue_shift(frame, hue_delta_deg):
    """
    Shift hue by hue_delta_deg degrees, keep S and V.
    """
    h, s, v = _rgb_to_hsv_np(frame)
    h = (h + hue_delta_deg) % 360.0
    return _hsv_to_rgb_np(h, s, v)

def _stable_rng(name, t, idx, features, fps, beat_response, latency_base_frames, seed=12345):
    """
    Deterministic RNG that stays constant over a segment.
    If beat timings are available, segments are groups of 'beat_response' beats.
    Otherwise, segments are groups of frames whose length shortens on strong beats.
    """
    beat_times = features.get("beat_times", None)
    if beat_times is not None and len(beat_times) > 0 and beat_response is not None:
        # which beat are we currently in?
        beat_idx = int(np.searchsorted(beat_times, t, side="right"))
        segment = int(beat_idx // max(1, int(beat_response)))
    else:
        # fallback: frame-grouping based on beat strength envelope
        beat_local = float(features.get("beat_strength", [0.0])[idx])
        group = max(1, int(latency_base_frames * (1.2 - 0.8 * beat_local)))
        segment = int(idx // group)

    key = (str(name), int(segment), int(seed))
    s = hash(key) & 0xFFFFFFFF
    return np.random.default_rng(s)

def _sample_normal_stable(name, mean, std_frac, t, idx, features, fps, beat_response, latency_base_frames, seed=12345, min_val=None, max_val=None):
    """
    Deterministic N(mean, std_frac*|mean|) sampler held constant within a segment.
    Segment definition: every N beats (beat_response) if beat times exist, else frame grouping.
    """
    mean = float(mean)
    std = abs(mean) * float(std_frac)
    rng = _stable_rng(name, t, idx, features, fps, beat_response, latency_base_frames, seed)
    val = float(rng.normal(loc=mean, scale=std if std > 0 else 0.0))
    if min_val is not None:
        val = max(min_val, val)
    if max_val is not None:
        val = min(max_val, val)
    return val

def _avg_complementary_color(frame):
    """
    Compute a complementary (inverted hue) colour based on the current frame's average hue.
    Returns an (R,G,B) uint8 tuple.
    """
    h, s, v = _rgb_to_hsv_np(frame)
    # average hue in degrees
    h_mean = float(np.mean(h))
    s_mean = float(np.mean(s))
    v_mean = float(np.mean(v))
    h_comp = (h_mean + 180.0) % 360.0
    rgb = _hsv_to_rgb_np(np.array(h_comp, dtype=np.float32),
                         np.array(s_mean, dtype=np.float32),
                         np.array(v_mean, dtype=np.float32))
    # scalar -> broadcast back then take first pixel
    if rgb.ndim == 3:
        r, g, b = [int(rgb[..., i].mean()) for i in range(3)]
    else:
        # rgb is a single pixel vector
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return (r, g, b)

def _blend_overlay(base_frame, overlay_img, alpha):
    """
    Alpha-composite a PIL overlay over the numpy base_frame.
    alpha 0..1 applied globally on overlay.
    """
    H, W, _ = base_frame.shape
    ov = overlay_img.convert("RGBA")
    # apply global alpha
    a = int(np.clip(alpha, 0.0, 1.0) * 255)
    r, g, b, _ = ov.split()
    ov = Image.merge("RGBA", (r, g, b, Image.new("L", ov.size, a)))
    base = Image.fromarray(base_frame.astype(np.uint8))
    base = base.convert("RGBA")
    base.alpha_composite(ov)
    return np.array(base.convert("RGB"))

def overlay_mandala(frame, color, complexity=8, thickness=2, opacity=0.4, seed=1234):
    """
    Draw a radial geometric pattern (circles + star polygons) using a complementary colour.
    complexity: number of spokes/wedges
    thickness: stroke width in pixels
    opacity: 0..1
    """
    H, W, _ = frame.shape
    rng = np.random.default_rng(seed)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(ov)

    cx, cy = W * 0.5, H * 0.5
    radius = min(W, H) * 0.45
    spokes = max(2, int(complexity))

    # concentric circles
    rings = max(2, int(2 + spokes // 2))
    for i in range(rings):
        r = radius * (i + 1) / rings
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw.ellipse(bbox, outline=color, width=int(thickness))

    # star polygon
    angles = np.linspace(0, 2 * np.pi, spokes, endpoint=False) + rng.uniform(0, np.pi / spokes)
    pts = [(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles]
    # connect every k-th point for a star look
    k = max(1, spokes // 3)
    for i in range(spokes):
        j = (i + k) % spokes
        draw.line([pts[i], pts[j]], fill=color, width=int(thickness))

    return _blend_overlay(frame, ov, opacity)

def overlay_tessellation(frame, color, cell=40, line_w=2, opacity=0.35):
    """
    Draw a grid tessellation overlay with adjustable cell size & line width.
    """
    H, W, _ = frame.shape
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(ov)

    # vertical lines
    x = 0
    while x < W:
        draw.line([(x, 0), (x, H)], fill=color, width=int(line_w))
        x += max(4, int(cell))
    # horizontal lines
    y = 0
    while y < H:
        draw.line([(0, y), (W, y)], fill=color, width=int(line_w))
        y += max(4, int(cell))

    return _blend_overlay(frame, ov, opacity)

def overlay_vector_field(frame, color, step=40, length=20, opacity=0.35):
    """
    Draw short line segments following image gradients sampled on a grid.
    Uses simple Sobel-like kernels without extra deps.
    """
    H, W, _ = frame.shape
    gray = (0.2126 * frame[..., 0] + 0.7152 * frame[..., 1] + 0.0722 * frame[..., 2]).astype(np.float32)

    # simple gradient (Sobel-lite)
    gx = (
        -gray[:, :-2] - 2 * gray[:, 1:-1] - gray[:, 2:]
        + gray[:, :-2].copy() * 0  # padding placeholder to keep shape hints
    )
    # rebuild gradients with padding to W,H
    # horizontal
    gx_full = np.zeros_like(gray)
    gx_full[:, 1:-1] = (-gray[:, :-2] + gray[:, 2:]) * 0.5
    # vertical
    gy_full = np.zeros_like(gray)
    gy_full[1:-1, :] = (-gray[:-2, :] + gray[2:, :]) * 0.5

    # sample grid and draw oriented strokes
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(ov)
    for y in range(step // 2, H, step):
        for x in range(step // 2, W, step):
            vx = gx_full[y, x]
            vy = gy_full[y, x]
            mag = np.hypot(vx, vy)
            if mag < 1e-3:
                continue
            vx /= mag
            vy /= mag
            dx = vx * length * 0.5
            dy = vy * length * 0.5
            draw.line([(x - dx, y - dy), (x + dx, y + dy)], fill=color, width=1)
    return _blend_overlay(frame, ov, opacity)

def overlay_spectral_rings(frame, color, bass_level, mid_level, high_level, opacity=0.4):
    """
    Draw concentric rings / polygon approximations whose sizes follow band levels.
    """
    H, W, _ = frame.shape
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(ov)
    cx, cy = W * 0.5, H * 0.5
    base_r = 0.15 * min(W, H)
    r_b = base_r * (1.0 + 1.5 * bass_level)
    r_m = base_r * (1.0 + 1.2 * mid_level)
    r_h = base_r * (1.0 + 1.0 * high_level)

    # three rings, thicker for bass
    draw.ellipse([cx - r_b, cy - r_b, cx + r_b, cy + r_b], outline=color, width=4)
    draw.ellipse([cx - r_m, cy - r_m, cx + r_m, cy + r_m], outline=color, width=2)
    # polygon for highs
    sides = max(3, int(3 + high_level * 9))
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    pts = [(cx + r_h * np.cos(a), cy + r_h * np.sin(a)) for a in ang]
    pts2 = pts + [pts[0]]
    draw.line(pts2, fill=color, width=2)

    return _blend_overlay(frame, ov, opacity)

def generate_frame(t, features, fps, imgs,
                   warp_strength, color_shift, repeat_period,
                   pixelation_max_block_size, scramble_max_frac,
                   randomize_images,
                   kaleido_on, kaleido_base_wedges, kaleido_rot_scale,
                   polar_on, polar_r_strength, polar_a_strength,
                   hue_on, hue_depth_deg,
                   # randomization controls
                   effect_randomness_pct, random_latency_frames, random_seed,
                   beat_response,
                   # overlays
                   ov_mandala_on, ov_mandala_complexity, ov_mandala_thick, ov_mandala_opacity,
                   ov_tess_on, ov_tess_cell, ov_tess_line_w, ov_tess_opacity,
                   ov_vfield_on, ov_vf_step, ov_vf_len, ov_vf_opacity,
                   ov_rings_on, ov_rings_opacity):
    """
    Build one frame at time t, with tempo-based hold / fade.

    Audio mapping:
    - bass  : colour tint.
    - highs : block pixelation size.
    - mids  : block scramble.
    - tempo : how long frames persist and how soft the transition is.
    """

    # safety: clamp index
    idx = int(t * fps)
    idx = min(idx, len(features["bass"]) - 1)

    # get tempo-driven parameters
    hold_sec, blend_soft = tempo_params(features, idx)

    # We treat the video timeline as 'keyframes' that advance every hold_sec.
    # Snap t to previous and next hold boundaries:
    if hold_sec <= 0:
        hold_sec = 1.0 / fps
    t_prev_key = np.floor(t / hold_sec) * hold_sec
    t_next_key = t_prev_key + hold_sec

    # alpha_key says where we are between these two keyframes (0..1)
    if t_next_key == t_prev_key:
        alpha_key = 0.0
    else:
        alpha_key = (t - t_prev_key) / (t_next_key - t_prev_key)
    alpha_key = np.clip(alpha_key, 0.0, 1.0)

    # soften alpha_key based on blend_soft:
    # higher softness => smoother ease between frames
    eased_alpha = alpha_key ** (1.0 / (1e-6 + (0.5 + 0.5 * blend_soft)))

    # pick which two images we're morphing between at this moment
    imgA_np, imgB_np = select_image_pair(
        t_prev_key,  # keyframed time drives which pair we see
        repeat_period,
        imgs,
        randomize_images=randomize_images
    )

    # For visual morph between imgA and imgB we still use musical repeat_period,
    # but driven by the snapped key time (t_prev_key) to give that "linger".
    alpha_morph = (t_prev_key % repeat_period) / repeat_period
    alpha_morph = np.clip(alpha_morph, 0.0, 1.0)

    # Base crossfade A->B at the snapped key (so content lags / holds),
    # but we also blend toward the *current* morph position a little,
    # controlled by eased_alpha, to avoid harsh steps.
    alpha_now = (t % repeat_period) / repeat_period
    alpha_now = np.clip(alpha_now, 0.0, 1.0)

    alpha_combined = (1.0 - eased_alpha) * alpha_morph + eased_alpha * alpha_now

    blended = (1 - alpha_combined) * imgA_np + alpha_combined * imgB_np
    frame = blended.astype(np.float32)

    # pull audio band levels at idx
    b_level   = float(features["bass"][idx])
    m_level   = float(features["mids"][idx])
    h_level   = float(features["highs"][idx])

    # local beat strength
    beat_local = float(features.get("beat_strength", [0.0])[idx])

    # randomization controls
    std_frac = float(effect_randomness_pct) / 100.0
    latency_base = int(max(1, random_latency_frames))
    seed = int(random_seed)

    # --- geometric layer: kaleidoscope and polar warp ---
    if kaleido_on:
        # base wedges respond to highs and beat
        wedges_mean = np.clip(kaleido_base_wedges + int(h_level * 10.0 * (1.0 + 0.5 * beat_local)), 2, 32)
        wedges_rand = _sample_normal_stable(
            "kaleido_wedges", wedges_mean, std_frac, t, idx, features, fps, beat_response, latency_base, seed,
            min_val=2, max_val=32
        )
        wedges = int(np.clip(round(wedges_rand), 2, 32))

        # rotation scale is randomized around the slider value
        rot_scale_rand = _sample_normal_stable(
            "kaleido_rot_scale", kaleido_rot_scale, std_frac, t, idx, features, fps, beat_response, latency_base, seed,
            min_val=0.0, max_val=1.0
        )
        rot = (rot_scale_rand * t) * (1.0 + 0.5 * m_level + 0.3 * beat_local)

        frame = kaleidoscope(frame.astype(np.uint8), wedges=wedges, rot=rot).astype(np.float32)

    if polar_on:
        r_strength = _sample_normal_stable(
            "polar_r_strength", polar_r_strength, std_frac, t, idx, features, fps, beat_response, latency_base, seed,
            min_val=0.0, max_val=1.0
        )
        a_strength = _sample_normal_stable(
            "polar_a_strength", polar_a_strength, std_frac, t, idx, features, fps, beat_response, latency_base, seed,
            min_val=0.0, max_val=1.0
        )
        r_amt = float(r_strength) * b_level
        a_amt = float(a_strength) * (m_level + 0.3 * beat_local)
        frame = polar_warp(frame.astype(np.uint8), r_amt=r_amt, a_amt=a_amt).astype(np.float32)

    # --- colour layer: hue shift tied to bass and a slow LFO ---
    if hue_on:
        hue_depth = _sample_normal_stable(
            "hue_depth_deg", hue_depth_deg, std_frac, t, idx, features, fps, beat_response, latency_base, seed,
            min_val=0.0, max_val=360.0
        )
        hue_delta = float(hue_depth) * (0.5 * b_level + 0.5 * np.sin(0.2 * t))
        frame = hue_shift(frame.astype(np.uint8), hue_delta_deg=hue_delta).astype(np.float32)

    # Bass: tint red
    frame[..., 0] = np.clip(frame[..., 0] * (1.0 + b_level * color_shift), 0, 255)

    # Treble: block pixelation (mosaic average per block)
    H, W, _ = frame.shape
    max_block = max(1, int(pixelation_max_block_size))
    block_size = 1 + int(h_level * (max_block - 1))
    if block_size < 1:
        block_size = 1

    if block_size > 1:
        pix_frame = frame.copy()
        for y0 in range(0, H, block_size):
            y1 = min(y0 + block_size, H)
            for x0 in range(0, W, block_size):
                x1 = min(x0 + block_size, W)
                block = frame[y0:y1, x0:x1, :]
                avg_col = block.mean(axis=(0,1), keepdims=True)
                pix_frame[y0:y1, x0:x1, :] = avg_col
        frame = pix_frame

    # Mids: scramble blocks after pixelation
    scramble_frac = float(scramble_max_frac) * m_level
    if scramble_frac > 0 and block_size > 1:
        num_blocks_y = int(np.ceil(H / block_size))
        num_blocks_x = int(np.ceil(W / block_size))
        total_blocks = num_blocks_y * num_blocks_x
        num_swap = int(total_blocks * scramble_frac)

        if num_swap > 1:
            y_blocks = np.random.randint(0, num_blocks_y, size=num_swap)
            x_blocks = np.random.randint(0, num_blocks_x, size=num_swap)
            y_blocks2 = np.random.randint(0, num_blocks_y, size=num_swap)
            x_blocks2 = np.random.randint(0, num_blocks_x, size=num_swap)

            frame_int = frame.astype(np.uint8)

            for i in range(num_swap):
                y0 = y_blocks[i] * block_size
                x0 = x_blocks[i] * block_size
                y1 = min(y0 + block_size, H)
                x1 = min(x0 + block_size, W)

                y0b = y_blocks2[i] * block_size
                x0b = x_blocks2[i] * block_size
                y1b = min(y0b + block_size, H)
                x1b = min(x0b + block_size, W)

                if (y1 - y0 == y1b - y0b) and (x1 - x0 == x1b - x0b):
                    tmp_block = frame_int[y0:y1, x0:x1].copy()
                    frame_int[y0:y1, x0:x1] = frame_int[y0b:y1b, x0b:x1b]
                    frame_int[y0b:y1b, x0b:x1b] = tmp_block

            frame = frame_int.astype(np.float32)
    elif scramble_frac > 0:
        num_pixels = H * W
        num_swap = int(num_pixels * scramble_frac)
        if num_swap > 1:
            ys = np.random.randint(0, H, size=num_swap)
            xs = np.random.randint(0, W, size=num_swap)
            ys2 = np.random.randint(0, H, size=num_swap)
            xs2 = np.random.randint(0, W, size=num_swap)
            frame_int = frame.astype(np.uint8)
            tmp_vals = frame_int[ys, xs].copy()
            frame_int[ys, xs] = frame_int[ys2, xs2]
            frame_int[ys2, xs2] = tmp_vals
            frame = frame_int.astype(np.float32)

    # --- overlays drawn last (on top) ---
    comp_color = _avg_complementary_color(frame)

    if ov_mandala_on:
        # randomness around user controls
        comp = max(2, int(_sample_normal_stable("ov_mandala_complexity",
                                                ov_mandala_complexity, std_frac, t, idx, features, fps, beat_response, latency_base, seed,
                                                min_val=2, max_val=64)))
        thick = max(1, int(_sample_normal_stable("ov_mandala_thick",
                                                 ov_mandala_thick, std_frac, t, idx, features, fps, beat_response, latency_base, seed,
                                                 min_val=1, max_val=20)))
        op = float(np.clip(_sample_normal_stable("ov_mandala_opacity",
                                                 ov_mandala_opacity, std_frac, t, idx, features, fps, beat_response, latency_base, seed,
                                                 min_val=0.0, max_val=1.0), 0.0, 1.0))
        frame = overlay_mandala(frame, comp_color, complexity=comp, thickness=thick, opacity=op, seed=seed)

    if ov_tess_on:
        cell = max(4, int(_sample_normal_stable("ov_tess_cell",
                                                ov_tess_cell, std_frac, t, idx, features, fps, beat_response, latency_base, seed,
                                                min_val=4, max_val=200)))
        lw = max(1, int(_sample_normal_stable("ov_tess_line_w",
                                              ov_tess_line_w, std_frac, t, idx, features, fps, beat_response, latency_base, seed,
                                              min_val=1, max_val=20)))
        op = float(np.clip(_sample_normal_stable("ov_tess_opacity",
                                                 ov_tess_opacity, std_frac, t, idx, features, fps, beat_response, latency_base, seed,
                                                 min_val=0.0, max_val=1.0), 0.0, 1.0))
        frame = overlay_tessellation(frame, comp_color, cell=cell, line_w=lw, opacity=op)

    if ov_vfield_on:
        step = max(8, int(_sample_normal_stable("ov_vf_step",
                                                ov_vf_step, std_frac, t, idx, features, fps, beat_response, latency_base, seed,
                                                min_val=8, max_val=200)))
        length = max(4, int(_sample_normal_stable("ov_vf_len",
                                                  ov_vf_len, std_frac, t, idx, features, fps, beat_response, latency_base, seed,
                                                  min_val=4, max_val=200)))
        op = float(np.clip(_sample_normal_stable("ov_vf_opacity",
                                                 ov_vf_opacity, std_frac, t, idx, features, fps, beat_response, latency_base, seed,
                                                 min_val=0.0, max_val=1.0), 0.0, 1.0))
        frame = overlay_vector_field(frame, comp_color, step=step, length=length, opacity=op)

    if ov_rings_on:
        op = float(np.clip(_sample_normal_stable("ov_rings_opacity",
                                                 ov_rings_opacity, std_frac, t, idx, features, fps, beat_response, latency_base, seed,
                                                 min_val=0.0, max_val=1.0), 0.0, 1.0))
        frame = overlay_spectral_rings(frame, comp_color, b_level, m_level, h_level, opacity=op)

    return frame.astype(np.uint8)

def render_video(y, sr, features, fps, imgs,
                 warp_strength, color_shift, repeat_period, resolution,
                 pixelation_max_block_size, scramble_max_frac,
                 randomize_images,
                 kaleido_on, kaleido_base_wedges, kaleido_rot_scale,
                 polar_on, polar_r_strength, polar_a_strength,
                 hue_on, hue_depth_deg,
                 effect_randomness_pct, random_latency_frames, random_seed,
                 beat_response,
                 ov_mandala_on, ov_mandala_complexity, ov_mandala_thick, ov_mandala_opacity,
                 ov_tess_on, ov_tess_cell, ov_tess_line_w, ov_tess_opacity,
                 ov_vfield_on, ov_vf_step, ov_vf_len, ov_vf_opacity,
                 ov_rings_on, ov_rings_opacity,
                 progress_callback=None):
    duration_s = len(y) / sr
    W, H = resolution

    def make_frame(t):
        frame = generate_frame(
            t,
            features,
            fps,
            imgs,
            warp_strength,
            color_shift,
            repeat_period,
            pixelation_max_block_size,
            scramble_max_frac,
            randomize_images,
            kaleido_on, kaleido_base_wedges, kaleido_rot_scale,
            polar_on, polar_r_strength, polar_a_strength,
            hue_on, hue_depth_deg,
            effect_randomness_pct, random_latency_frames, random_seed,
            beat_response,
            ov_mandala_on, ov_mandala_complexity, ov_mandala_thick, ov_mandala_opacity,
            ov_tess_on, ov_tess_cell, ov_tess_line_w, ov_tess_opacity,
            ov_vfield_on, ov_vf_step, ov_vf_len, ov_vf_opacity,
            ov_rings_on, ov_rings_opacity
        )
        # ensure final frame is at output resolution W x H
        frame_resized = np.array(Image.fromarray(frame).resize((W, H)))
        return frame_resized

    clip = mpy.VideoClip(make_frame, duration=duration_s)

    # Write full audio to a temp WAV to preserve exact SR for ffmpeg
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as aw:
        sf.write(aw.name, y.astype(np.float32), sr)
        audio_clip = AudioFileClip(aw.name)
    clip = clip.set_audio(audio_clip)

    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        # Older MoviePy versions do not support custom callbacks / progress_bar args.
        # We'll just write the file, then report completion.
        clip.write_videofile(
            tmp.name,
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            verbose=False,
            logger=None
        )
        tmp.seek(0)

        # let the UI know we're done
        if progress_callback is not None:
            progress_callback(1.0)

        return tmp.read()

def render_preview_clip(y, sr,
                        features, fps, imgs,
                        warp_strength, color_shift,
                        repeat_period,
                        start_time, duration_s,
                        preview_res=(320,180), preview_fps=15,
                        pixelation_max_block_size=10,
                        scramble_max_frac=0.1,
                        randomize_images=True,
                        kaleido_on=True, kaleido_base_wedges=6, kaleido_rot_scale=0.05,
                        polar_on=True, polar_r_strength=0.12, polar_a_strength=0.20,
                        hue_on=True, hue_depth_deg=60,
                        effect_randomness_pct=0.0, random_latency_frames=12, random_seed=12345,
                        beat_response=1,
                        ov_mandala_on=True, ov_mandala_complexity=8, ov_mandala_thick=2, ov_mandala_opacity=0.4,
                        ov_tess_on=False, ov_tess_cell=40, ov_tess_line_w=2, ov_tess_opacity=0.35,
                        ov_vfield_on=False, ov_vf_step=40, ov_vf_len=20, ov_vf_opacity=0.35,
                        ov_rings_on=False, ov_rings_opacity=0.4,
                        progress_callback=None):
    W, H = preview_res

    def make_frame_local(t_local):
        t_abs = start_time + t_local
        frame = generate_frame(
            t_abs,
            features,
            fps,
            imgs,
            warp_strength,
            color_shift,
            repeat_period,
            pixelation_max_block_size,
            scramble_max_frac,
            randomize_images,
            kaleido_on, kaleido_base_wedges, kaleido_rot_scale,
            polar_on, polar_r_strength, polar_a_strength,
            hue_on, hue_depth_deg,
            effect_randomness_pct, random_latency_frames, random_seed,
            beat_response,
            ov_mandala_on, ov_mandala_complexity, ov_mandala_thick, ov_mandala_opacity,
            ov_tess_on, ov_tess_cell, ov_tess_line_w, ov_tess_opacity,
            ov_vfield_on, ov_vf_step, ov_vf_len, ov_vf_opacity,
            ov_rings_on, ov_rings_opacity
        )
        frame_resized = np.array(Image.fromarray(frame).resize((W, H)))
        return frame_resized

    clip = mpy.VideoClip(make_frame_local, duration=duration_s)

    # grab matching audio slice
    start_sample = int(start_time * sr)
    end_sample = int((start_time + duration_s) * sr)
    end_sample = min(end_sample, len(y))
    audio_slice = y[start_sample:end_sample]

    # Force exact length to match preview duration
    target_samples = int(round(duration_s * sr))
    cur = audio_slice.shape[0]
    if cur < target_samples:
        pad = np.zeros((target_samples - cur,), dtype=audio_slice.dtype)
        audio_slice = np.concatenate([audio_slice, pad], axis=0)
    elif cur > target_samples:
        audio_slice = audio_slice[:target_samples]

    if audio_slice.size > 0:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as awprev:
            sf.write(awprev.name, audio_slice.astype(np.float32), sr)
            preview_audio = AudioFileClip(awprev.name)
        clip = clip.set_audio(preview_audio)

    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        clip.write_videofile(
            tmp.name,
            fps=preview_fps,
            codec="libx264",
            audio_codec="aac",
            verbose=False,
            logger=None
        )
        tmp.seek(0)

        if progress_callback is not None:
            progress_callback(1.0)

        return tmp.read()

def download_link(data_bytes, filename, label):
    b64 = base64.b64encode(data_bytes).decode()
    href = f'<a href="data:video/mp4;base64,{b64}" download="{filename}">{label}</a>'
    return href

# -----------------------
# Streamlit UI
# -----------------------

st.title("Audio reactive video builder")

st.sidebar.header("Inputs")

audio_file = st.sidebar.file_uploader("Upload audio", type=["wav","mp3"])
img_files  = st.sidebar.file_uploader("Upload images", type=["png","jpg","jpeg"], accept_multiple_files=True)

fps = st.sidebar.slider("Video FPS", 15, 60, 15)
width = st.sidebar.slider("Width (px)", 320, 1920, 320)
height = st.sidebar.slider("Height (px)", 240, 1080, 240)

warp_strength = st.sidebar.slider("Warp strength", 0.0, 2.0, 0.5, 0.01)
color_shift   = st.sidebar.slider("Color shift from bass", 0.0, 2.0, 0.8, 0.01)

pixelation_max_block_size = st.sidebar.slider(
    "Max pixel block size (px)",
    min_value=1,
    max_value=100,
    value=50,
    step=1,
    help="At max treble, each block will be this many pixels wide/high."
)

scramble_max_frac = st.sidebar.slider(
    "Scramble max fraction",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.01,
    help="Max fraction of pixels to scramble at peak mids."
)

st.sidebar.write("Morph timing etc will become editable per image pair later.")

st.sidebar.markdown("---")
st.sidebar.subheader("Geometric and colour effects")

kaleido_on = st.sidebar.checkbox("Enable kaleidoscope", value=True)
kaleido_base_wedges = st.sidebar.slider("Kaleidoscope base wedges", 2, 16, 6)
kaleido_rot_scale = st.sidebar.slider("Kaleidoscope rotation scale", 0.0, 0.2, 0.05, 0.005)

polar_on = st.sidebar.checkbox("Enable polar warp", value=True)
polar_r_strength = st.sidebar.slider("Polar radial strength", 0.0, 0.6, 0.12, 0.01)
polar_a_strength = st.sidebar.slider("Polar angular strength", 0.0, 0.6, 0.20, 0.01)

hue_on = st.sidebar.checkbox("Enable hue shift", value=True)
hue_depth_deg = st.sidebar.slider("Hue shift depth (degrees)", 0, 180, 60, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("Randomization")
effect_randomness_pct = st.sidebar.slider(
    "Effect randomness (std % of value)", 0, 100, 20, 1,
    help="Normal-noise std as a percent of the current control value."
)
random_latency_frames = st.sidebar.slider(
    "Randomization latency (frames)", 1, 60, 12, 1,
    help="How many frames a randomized value persists. Strong beats shorten this."
)
random_seed = st.sidebar.number_input(
    "Random seed", min_value=0, max_value=10**9, value=12345, step=1
)
beat_response = st.sidebar.slider(
    "Beat response (beats per change)", 1, 128, 2, 1,
    help="1 = change every beat, 8 = change every 8th beat."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Overlays")

ov_mandala_on = st.sidebar.checkbox("Overlay: mandala (complementary colour)", value=True)
ov_mandala_complexity = st.sidebar.slider("Mandala complexity (spokes)", 2, 64, 12, 1)
ov_mandala_thick = st.sidebar.slider("Mandala stroke width", 1, 12, 2, 1)
ov_mandala_opacity = st.sidebar.slider("Mandala opacity", 0.0, 1.0, 0.4, 0.01)

ov_tess_on = st.sidebar.checkbox("Overlay: tessellation grid", value=False)
ov_tess_cell = st.sidebar.slider("Tessellation cell (px)", 4, 200, 48, 2)
ov_tess_line_w = st.sidebar.slider("Tessellation line width", 1, 20, 2, 1)
ov_tess_opacity = st.sidebar.slider("Tessellation opacity", 0.0, 1.0, 0.35, 0.01)

ov_vfield_on = st.sidebar.checkbox("Overlay: vector field lines", value=False)
ov_vf_step = st.sidebar.slider("Vector field grid step (px)", 8, 200, 40, 2)
ov_vf_len = st.sidebar.slider("Vector field line length (px)", 4, 200, 20, 2)
ov_vf_opacity = st.sidebar.slider("Vector field opacity", 0.0, 1.0, 0.35, 0.01)

ov_rings_on = st.sidebar.checkbox("Overlay: spectral rings/polygons", value=False)
ov_rings_opacity = st.sidebar.slider("Spectral rings opacity", 0.0, 1.0, 0.4, 0.01)

if audio_file and img_files and len(img_files) >= 1:
    # Load audio
    y, sr = load_audio(audio_file.read())
    st.write(f"Audio length: {len(y)/sr:.1f} s, sample rate {sr} Hz")

    # Analyse audio
    features = analyze_audio(y, sr, fps)

    import pandas as pd

    st.subheader("Audio feature preview")

    # Build a DataFrame indexed by time, so time becomes the x axis
    feature_df = pd.DataFrame({
        "bass": features["bass"],
        "mids": features["mids"],
        "highs": features["highs"],
        "amp (loudness)": features["amp"],
    }, index=features["times"])

    feature_df.index.name = "time (s)"

    st.line_chart(feature_df)

    audio_len_s = len(y) / sr

    repeat_period = st.sidebar.number_input(
        "Repeat period (s)",
        min_value=1.0,
        max_value=60.0,
        value=10.0,
        step=1.0
    )

    start_time = st.sidebar.slider(
        "Preview start in audio (s)",
        min_value=0.0,
        max_value=max(0.0, float(audio_len_s - repeat_period)),
        value=0.0,
        step=0.5
    )

    # Load all uploaded images
    pil_imgs = [Image.open(f).convert("RGB") for f in img_files]

    # Force all to the size of the first image for consistent blending
    base_w, base_h = pil_imgs[0].size
    pil_imgs_resized = [im.resize((base_w, base_h)) for im in pil_imgs]

    # Convert to numpy arrays
    imgs = [np.array(im) for im in pil_imgs_resized]

    # UI toggle: random or sequential cycling
    randomize_images = st.sidebar.checkbox(
        "Randomize image pairs each period",
        value=True
    )

    

    st.subheader("Period preview")
    st.write(f"From {start_time:.1f} s to {(start_time+repeat_period):.1f} s")
    if st.button("Preview period clip"):
        preview_progress = st.progress(0.0)
        def preview_cb(p):
            preview_progress.progress(p)
        clip_bytes = render_preview_clip(
            y, sr,
            features,
            fps,
            imgs,
            warp_strength,
            color_shift,
            repeat_period,
            start_time,
            duration_s=repeat_period,
            preview_res=(320,180),
            preview_fps=15,
            pixelation_max_block_size=pixelation_max_block_size,
            scramble_max_frac=scramble_max_frac,
            randomize_images=randomize_images,
            kaleido_on=kaleido_on, kaleido_base_wedges=kaleido_base_wedges, kaleido_rot_scale=kaleido_rot_scale,
            polar_on=polar_on, polar_r_strength=polar_r_strength, polar_a_strength=polar_a_strength,
            hue_on=hue_on, hue_depth_deg=hue_depth_deg,
            effect_randomness_pct=effect_randomness_pct,
            random_latency_frames=random_latency_frames,
            random_seed=random_seed,
            beat_response=beat_response,
            ov_mandala_on=ov_mandala_on, ov_mandala_complexity=ov_mandala_complexity, ov_mandala_thick=ov_mandala_thick, ov_mandala_opacity=ov_mandala_opacity,
            ov_tess_on=ov_tess_on, ov_tess_cell=ov_tess_cell, ov_tess_line_w=ov_tess_line_w, ov_tess_opacity=ov_tess_opacity,
            ov_vfield_on=ov_vfield_on, ov_vf_step=ov_vf_step, ov_vf_len=ov_vf_len, ov_vf_opacity=ov_vf_opacity,
            ov_rings_on=ov_rings_on, ov_rings_opacity=ov_rings_opacity,
            progress_callback=preview_cb
        )
        st.video(clip_bytes)

    st.subheader("Final render")
    if st.button("Render full video"):
        final_progress = st.progress(0.0)
        def final_cb(p):
            final_progress.progress(p)
        video_bytes = render_video(
            y, sr, features, fps,
            imgs,
            warp_strength,
            color_shift,
            repeat_period,
            (width, height),
            pixelation_max_block_size,
            scramble_max_frac,
            randomize_images,
            kaleido_on, kaleido_base_wedges, kaleido_rot_scale,
            polar_on, polar_r_strength, polar_a_strength,
            hue_on, hue_depth_deg,
            effect_randomness_pct, random_latency_frames, random_seed,
            beat_response,
            ov_mandala_on, ov_mandala_complexity, ov_mandala_thick, ov_mandala_opacity,
            ov_tess_on, ov_tess_cell, ov_tess_line_w, ov_tess_opacity,
            ov_vfield_on, ov_vf_step, ov_vf_len, ov_vf_opacity,
            ov_rings_on, ov_rings_opacity,
            progress_callback=final_cb
        )
        st.video(video_bytes)
        st.markdown(download_link(video_bytes, "output.mp4", "Download MP4"),
                    unsafe_allow_html=True)
else:
    st.info("Upload audio and at least one image to begin.")# trigger rebuild
