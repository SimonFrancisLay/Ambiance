import streamlit as st
import numpy as np
import tempfile
import librosa
import moviepy.editor as mpy
from PIL import Image
from io import BytesIO
import base64
from moviepy.audio.AudioClip import AudioArrayClip

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
        # beat_strength length should match amp length
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

def generate_frame(t, features, fps, imgs,
                   warp_strength, color_shift, repeat_period,
                   pixelation_max_block_size, scramble_max_frac,
                   randomize_images):
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

    return frame.astype(np.uint8)

def render_video(y, sr, features, fps, imgs,
                 warp_strength, color_shift, repeat_period, resolution,
                 pixelation_max_block_size, scramble_max_frac,
                 randomize_images,
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
            randomize_images
        )
        # ensure final frame is at output resolution W x H
        frame_resized = np.array(Image.fromarray(frame).resize((W, H)))
        return frame_resized

    clip = mpy.VideoClip(make_frame, duration=duration_s)

    audio_clip = AudioArrayClip(y.reshape(-1, 1), fps=sr)
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
            randomize_images
        )
        frame_resized = np.array(Image.fromarray(frame).resize((W, H)))
        return frame_resized

    clip = mpy.VideoClip(make_frame_local, duration=duration_s)

    # grab matching audio slice
    start_sample = int(start_time * sr)
    end_sample = int((start_time + duration_s) * sr)
    end_sample = min(end_sample, len(y))
    audio_slice = y[start_sample:end_sample]

    if audio_slice.size > 0:
        preview_audio = AudioArrayClip(audio_slice.reshape(-1, 1), fps=sr)
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
            progress_callback=final_cb
        )
        st.video(video_bytes)
        st.markdown(download_link(video_bytes, "output.mp4", "Download MP4"),
                    unsafe_allow_html=True)
else:
    st.info("Upload audio and at least one image to begin.")