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

    # normalise to 0..1 for convenience
    def norm(v):
        v = np.array(v)
        v = v / (np.max(v) + 1e-9)
        return v

    features = {
        "amp": norm(amp),
        "bass": norm(bass),
        "mids": norm(mids),
        "highs": norm(highs),
        "times": frame_starts[:len(amp)] / sr
    }
    return features

def generate_frame(t, features, fps, imgA_np, imgB_np,
                   warp_strength, color_shift, repeat_period,
                   pixelation_max_block_size, scramble_max_frac):
    """
    Build one frame at time t.

    Audio mapping:
    - bass  : drives colour tint (red boost for now).
    - highs : drives pixelation (high highs -> blocky).
    - mids  : drives random scramble (<=10 percent of pixels swapped).

    Visual flow:
    1. Determine alpha inside the morph loop (repeat_period).
    2. Crossfade imgA -> imgB.
    3. Apply colour tint from bass.
    4. Apply pixelation from highs.
    5. Apply scramble from mids.
    """

    # safety: clamp fps index to available features
    idx = int(t * fps)
    idx = min(idx, len(features["bass"]) - 1)

    # 1. morph factor alpha cycles over repeat_period seconds
    alpha = (t % repeat_period) / repeat_period
    alpha = np.clip(alpha, 0.0, 1.0)

    # base crossfade between the 2 images
    blended = (1 - alpha) * imgA_np + alpha * imgB_np  # float math
    frame = blended.astype(np.float32)

    # pull feature levels
    b_level   = float(features["bass"][idx])
    m_level   = float(features["mids"][idx])
    h_level   = float(features["highs"][idx])

    # 2. Colour tint from bass (same as before, just using frame var)
    # boost red channel proportional to bass * color_shift
    frame[..., 0] = np.clip(frame[..., 0] * (1.0 + b_level * color_shift), 0, 255)

    # 3. Pixelation from highs.
    # We compute a downsample factor between 1 (no pixelation) and pixelation_max_block_size (very blocky).
    # h_level near 1.0 means use max block size.
    H, W, _ = frame.shape
    max_factor = max(1, int(pixelation_max_block_size))
    # linear interpolate: factor = 1 .. max_factor
    factor = 1 + int(h_level * (max_factor - 1))

    # effective small size (W/factor, H/factor)
    small_w = max(1, int(W / factor))
    small_h = max(1, int(H / factor))

    if small_w < W or small_h < H:
        pil_tmp = Image.fromarray(frame.astype(np.uint8))
        pil_small = pil_tmp.resize((small_w, small_h), resample=Image.NEAREST)
        pil_blocky = pil_small.resize((W, H), resample=Image.NEAREST)
        frame = np.array(pil_blocky).astype(np.float32)

    # 4. Scramble from mids.
    # We will randomly shuffle up to scramble_max_frac of pixels.
    # The exact fraction is 0..scramble_max_frac scaled by m_level.
    scramble_frac = float(scramble_max_frac) * m_level
    if scramble_frac > 0:
        num_pixels = H * W
        num_swap = int(num_pixels * scramble_frac)

        if num_swap > 1:
            # pick indices to swap
            ys = np.random.randint(0, H, size=num_swap)
            xs = np.random.randint(0, W, size=num_swap)
            ys2 = np.random.randint(0, H, size=num_swap)
            xs2 = np.random.randint(0, W, size=num_swap)

            # perform swap on a copy
            frame_int = frame.astype(np.uint8)
            tmp_vals = frame_int[ys, xs].copy()
            frame_int[ys, xs] = frame_int[ys2, xs2]
            frame_int[ys2, xs2] = tmp_vals
            frame = frame_int.astype(np.float32)

    # TODO: Apply smooth spatial warp using warp_strength * b_level (future step).

    return frame.astype(np.uint8)

def render_video(y, sr, features, fps, imgA_np, imgB_np,
                 warp_strength, color_shift, repeat_period, resolution,
                 pixelation_max_block_size, scramble_max_frac, progress_callback=None):
    duration_s = len(y) / sr
    W, H = resolution

    def make_frame(t):
        frame = generate_frame(
            t,
            features,
            fps,
            np.array(Image.fromarray(imgA_np).resize((W, H))),
            np.array(Image.fromarray(imgB_np).resize((W, H))),
            warp_strength,
            color_shift,
            repeat_period,
            pixelation_max_block_size,
            scramble_max_frac
        )
        return frame

    clip = mpy.VideoClip(make_frame, duration=duration_s)

    audio_clip = AudioArrayClip(y.reshape(-1, 1), fps=sr)
    clip = clip.set_audio(audio_clip)

    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        # When writing, MoviePy itself iterates frames. We can hook a callback for UI progress.
        def _prog_cb(current_time):
            if progress_callback is not None and duration_s > 0:
                progress_callback(min(current_time / duration_s, 1.0))

        clip.write_videofile(
            tmp.name,
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            verbose=False,
            logger=None,
            progress_bar=False,
            write_logfile=False,
            callbacks=[("progress_bar", _prog_cb)]
        )
        tmp.seek(0)
        return tmp.read()

def render_preview_clip(features, fps, imgA_np, imgB_np,
                        warp_strength, color_shift,
                        repeat_period,
                        start_time, duration_s,
                        preview_res=(320,180), preview_fps=15,
                        pixelation_max_block_size=10,
                        scramble_max_frac=0.1,
                        progress_callback=None):
    W, H = preview_res

    def make_frame_local(t_local):
        t_abs = start_time + t_local
        frame = generate_frame(
            t_abs,
            features,
            fps,
            np.array(Image.fromarray(imgA_np).resize((W, H))),
            np.array(Image.fromarray(imgB_np).resize((W, H))),
            warp_strength,
            color_shift,
            repeat_period,
            pixelation_max_block_size,
            scramble_max_frac
        )
        return frame

    clip = mpy.VideoClip(make_frame_local, duration=duration_s)

    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        def _prog_cb(current_time):
            if progress_callback is not None and duration_s > 0:
                progress_callback(min(current_time / duration_s, 1.0))

        clip.write_videofile(
            tmp.name,
            fps=preview_fps,
            codec="libx264",
            audio=False,
            verbose=False,
            logger=None,
            progress_bar=False,
            write_logfile=False,
            callbacks=[("progress_bar", _prog_cb)]
        )
        tmp.seek(0)
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
    "Pixelation strength (bigger = blockier)",
    min_value=1,
    max_value=20,
    value=10,
    step=1,
    help="At max highs, we downsample by this factor. 10 on a 320x240 frame -> 32x24 blocks."
)

scramble_max_frac = st.sidebar.slider(
    "Scramble max fraction",
    min_value=0.0,
    max_value=0.5,
    value=0.1,
    step=0.01,
    help="Max fraction of pixels to scramble at peak mids."
)

st.sidebar.write("Morph timing etc will become editable per image pair later.")

if audio_file and img_files and len(img_files) >= 2:
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

    # Take first two images for now
    imgA = Image.open(img_files[0]).convert("RGB")
    imgB = Image.open(img_files[1]).convert("RGB")

    # force same size
    imgB = imgB.resize(imgA.size)

    imgA_np = np.array(imgA)
    imgB_np = np.array(imgB)

    

    st.subheader("Period preview")
    st.write(f"From {start_time:.1f} s to {(start_time+repeat_period):.1f} s")
    if st.button("Preview period clip"):
        preview_progress = st.progress(0.0)
        def preview_cb(p):
            preview_progress.progress(p)
        clip_bytes = render_preview_clip(
            features,
            fps,
            imgA_np,
            imgB_np,
            warp_strength,
            color_shift,
            repeat_period,
            start_time,
            duration_s=repeat_period,
            preview_res=(320,180),
            preview_fps=15,
            pixelation_max_block_size=pixelation_max_block_size,
            scramble_max_frac=scramble_max_frac,
            progress_callback=preview_cb
        )
        preview_progress.progress(1.0)
        st.video(clip_bytes)

    st.subheader("Final render")
    if st.button("Render full video"):
        final_progress = st.progress(0.0)
        def final_cb(p):
            final_progress.progress(p)
        video_bytes = render_video(
            y, sr, features, fps,
            imgA_np, imgB_np,
            warp_strength,
            color_shift,
            repeat_period,
            (width, height),
            pixelation_max_block_size,
            scramble_max_frac,
            progress_callback=final_cb
        )
        final_progress.progress(1.0)
        st.video(video_bytes)
        st.markdown(download_link(video_bytes, "output.mp4", "Download MP4"),
                    unsafe_allow_html=True)
else:
    st.info("Upload audio and at least two images to begin.")