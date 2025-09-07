# app.py
# Interactive Comic Generator (Gemini 2.5 Flash Image Preview)
# - Latest SDK imports (google-genai, elevenlabs, moviepy v2)
# - No FAL usage
# - width='stretch' (no use_container_width)
# - Page-flip SFX after each slideshow panel
# - Panel animations (Page Flip 3D, Comic Pop, Ken Burns)
# - FIX: removed nested f-strings that caused SyntaxError

from __future__ import annotations

import os
import re
import time
import base64
from io import BytesIO
from html import escape as html_escape
from pathlib import Path

import streamlit as st
from streamlit.components.v1 import html as st_html
from dotenv import load_dotenv
from PIL import Image, ImageFont, ImageDraw

# Google GenAI unified SDK
from google import genai
from google.genai import types

# ElevenLabs SDK (optional narration/SFX)
from elevenlabs.client import ElevenLabs
from elevenlabs import save

# MoviePy v2.x + imageio-ffmpeg
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips
import imageio_ffmpeg

# -----------------------------
# App setup
# -----------------------------
st.set_page_config(page_title="Interactive Comic Generator 🍌", layout="wide")
st.title("Interactive Comic Generator 🍌 (Gemini 2.5 Flash Image Preview)")
load_dotenv()


# Prefer Streamlit secrets; fall back to env vars
def _gemini_key():
    try:
        return st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        return None


# Gemini Client (Developer API)
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error(
        "No Gemini API key found. Set `GEMINI_API_KEY` in Streamlit Secrets (or env var)."
    )
    st.stop()


http_opts = types.HttpOptions(api_version="v1alpha")  # preview image model
client = genai.Client(api_key=api_key, http_options=http_opts)
MODEL_NAME = "gemini-2.5-flash-image-preview"

# ElevenLabs (optional)
eleven_api_key = os.getenv("ELEVENLABS_API_KEY") or st.text_input(
    "Enter ElevenLabs API Key (optional for narration/SFX):", type="password"
)
eleven_client: Optional[ElevenLabs] = (
    ElevenLabs(api_key=eleven_api_key) if eleven_api_key else None
)

# Ensure MoviePy finds FFmpeg from imageio-ffmpeg wheel
os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

# -----------------------------
# Session defaults (set BEFORE widgets)
# -----------------------------
st.session_state.setdefault("story_prompt", "")
st.session_state.setdefault("num_panels", 4)
st.session_state.setdefault("style", "Manga")
st.session_state.setdefault("animation_style", "Page Flip (3D)")
st.session_state.setdefault("add_narration", False)
st.session_state.setdefault(
    "add_page_flip", bool(eleven_client)
)  # default SFX on if ElevenLabs present
st.session_state.setdefault("demo_mode", False)
st.session_state.setdefault("demo_has_run", False)

for k, v in {
    "panels": [],
    "images": [],
    "dialogues": [],
    "audio_files": [],
    "page_flip_b64": None,
}.items():
    st.session_state.setdefault(k, v)

# -----------------------------
# DEMO MODE toggle (before other widgets)
# -----------------------------
demo_mode = st.checkbox(
    "🎬 Demo Mode (auto-run a great example)",
    value=st.session_state["demo_mode"],
    help="Autofills a strong prompt, sets 4 panels, Page Flip (3D)+SFX, then auto-generates and plays slideshow once.",
    key="demo_mode",
)
if not demo_mode:
    st.session_state["demo_has_run"] = False
if demo_mode:
    if not st.session_state["story_prompt"]:
        st.session_state["story_prompt"] = (
            "A noir cat burglar in neon Tokyo is chased by drone cops across rooftops."
        )
    st.session_state["num_panels"] = 4
    st.session_state["style"] = "Manga"
    st.session_state["animation_style"] = "Page Flip (3D)"
    if eleven_client:
        st.session_state["add_page_flip"] = True  # narration still optional

# -----------------------------
# Quick Prompts (before text_area)
# -----------------------------
st.caption("Quick prompts")
c1, c2, c3, c4 = st.columns(4)
if c1.button("Neon noir chase"):
    st.session_state["story_prompt"] = (
        "A noir cat burglar in neon Tokyo is chased by drone cops across rooftops."
    )
if c2.button("Fantasy quest"):
    st.session_state["story_prompt"] = (
        "A young mage and a grumpy knight quest through a living forest to rescue a phoenix egg."
    )
if c3.button("Manga mecha duel"):
    st.session_state["story_prompt"] = (
        "Two rival pilots battle in towering mechs above a flooded coastal megacity at dusk."
    )
if c4.button("Slice-of-life sci-fi"):
    st.session_state["story_prompt"] = (
        "A coffee-obsessed astronaut on a long mission improvises a new brew with zero-G hijinks."
    )

# -----------------------------
# Widgets bound to state keys
# -----------------------------
st.text_area(
    "Enter your story prompt:",
    key="story_prompt",
    placeholder="e.g., A robot detective solves a mystery in a futuristic city.",
)
st.slider("Number of panels:", 3, 8, key="num_panels")
st.selectbox(
    "Art style:",
    ["Cyberpunk", "Cartoon", "Realistic", "Manga", "Photorealistic"],
    key="style",
)
st.selectbox(
    "Panel animation style",
    ["Page Flip (3D)", "Comic Pop", "Ken Burns", "None"],
    key="animation_style",
)

if eleven_client:
    st.checkbox("Add Voice Narration (ElevenLabs)", key="add_narration")
    st.checkbox("Add Page Flip Sound (ElevenLabs)", key="add_page_flip")
else:
    st.session_state["add_narration"] = False
    st.session_state["add_page_flip"] = False

# Local copies
story_prompt_val = (st.session_state["story_prompt"] or "").strip()
num_panels = int(st.session_state["num_panels"])
style = st.session_state["style"]
animation_style = st.session_state["animation_style"]
add_narration = bool(st.session_state["add_narration"])
add_page_flip = bool(st.session_state["add_page_flip"])


# -----------------------------
# CSS / Anim helpers
# -----------------------------
def _animation_css() -> str:
    return """
      <style>
        .card { position: relative; width: 100%; will-change: transform, filter; }
        .img { width: 100%; height: auto; border-radius: 14px; box-shadow: 0 8px 30px rgba(0,0,0,.15); display:block; }

        /* Speech bubble overlay */
        .bubble {
          position: absolute; left: 12px; right: 12px; bottom: 12px;
          background: #fff; border: 3px solid #000; border-radius: 16px;
          padding: 10px 14px; box-shadow: 0 4px 0 #000, 0 8px 24px rgba(0,0,0,.20);
          font-weight: 900; line-height: 1.25;
          font-family: Impact, "Anton", "Bangers", "Comic Sans MS", system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
          letter-spacing: 0.3px; color: #000; text-shadow: 0 1px 0 rgba(255,255,255,0.6);
        }
        .bubble .label {
          display: block; font-size: 0.8em; letter-spacing: 0.06em; opacity: 0.7; margin-bottom: 2px;
          text-transform: uppercase;
        }
        .bubble .text { font-size: clamp(14px, 1.9vw, 20px); word-break: break-word; }
        .bubble .tail {
          position: absolute; left: 32px; bottom: -14px; width: 0; height: 0;
          border-left: 14px solid #000; border-top: 14px solid transparent;
        }
        .bubble .tail::after {
          content:""; position: absolute; left: -12px; top: -12px; width: 0; height: 0;
          border-left: 12px solid #fff; border-top: 12px solid transparent;
        }

        /* Page Flip (3D) */
        .anim-flip-start { transform-style: preserve-3d; perspective: 1200px; }
        .anim-flip { animation: flipIn .5s ease-out both; transform-origin: left center; }
        @keyframes flipIn {
          0% { transform: rotateY(-85deg); filter: brightness(0.9) saturate(0.95); }
          60% { transform: rotateY(8deg); }
          100% { transform: rotateY(0deg); }
        }

        /* Comic Pop */
        .anim-pop { animation: popIn .42s cubic-bezier(.2,.9,.2,1.1) both; }
        @keyframes popIn { 0% { transform: scale(.82); } 70% { transform: scale(1.04); } 100% { transform: scale(1); } }

        /* Ken Burns */
        .anim-kenburns { animation: ken 4s ease-out both; }
        @keyframes ken { 0% { transform: scale(1.02) translate3d(0,6px,0); } 100% { transform: scale(1.07) translate3d(0,-6px,0); } }
      </style>
    """


def _anim_class(style_name: str) -> tuple[str, bool]:
    if style_name == "Page Flip (3D)":
        return ("anim-flip", True)
    if style_name == "Comic Pop":
        return ("anim-pop", False)
    if style_name == "Ken Burns":
        return ("anim-kenburns", False)
    return ("", False)


def render_animated_panel(
    img: Image.Image, idx: int, caption: str, sfx_b64: Optional[str], style_name: str
):
    buf = BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    anim_class, needs_wrap = _anim_class(style_name)
    wrap_class = "anim-flip-start" if needs_wrap else ""
    js_audio = ""
    if sfx_b64:
        js_audio = f"\n                new Audio('data:audio/mp3;base64,{sfx_b64}').play().catch(()=>{{}});\n"
    safe_caption = html_escape(caption)
    panel_id = f"panel-{idx}-{int(time.time()*1000)}"
    html = (
        _animation_css()
        + f"""
      <div id="{panel_id}" class="card {wrap_class}">
        <img class="img" alt="Panel {idx+1}" src="data:image/png;base64,{img_b64}" />
        <div class="bubble">
          <span class="label">Panel {idx+1}</span>
          <div class="text">{safe_caption}</div>
          <span class="tail"></span>
        </div>
      </div>
      <script>
        (function() {{
          const el = document.getElementById("{panel_id}");
          const img = el.querySelector('.img');
          const obs = new IntersectionObserver((ents, o) => {{
            ents.forEach(e => {{
              if (e.isIntersecting) {{
                img.classList.add("{anim_class}");
                o.disconnect();
                {js_audio}
              }}
            }});
          }}, {{ threshold: 0.5 }});
          obs.observe(el);
        }})();
      </script>
    """
    )
    st_html(html, height=img.height + 160, scrolling=False)


def render_slideshow_panel(img: Image.Image, idx: int, caption: str, style_name: str):
    buf = BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    anim_class, needs_wrap = _anim_class(style_name)
    wrap_class = "anim-flip-start" if needs_wrap else ""
    safe_caption = html_escape(caption)
    panel_id = f"slide-{idx}-{int(time.time()*1000)}"
    html = (
        _animation_css()
        + f"""
      <div id="{panel_id}" class="card {wrap_class}">
        <img class="img" alt="Panel {idx+1}" src="data:image/png;base64,{img_b64}" />
        <div class="bubble">
          <span class="label">Panel {idx+1}</span>
          <div class="text">{safe_caption}</div>
          <span class="tail"></span>
        </div>
      </div>
      <script>
        (function() {{
          const img = document.querySelector('#{panel_id} .img');
          requestAnimationFrame(() => img.classList.add("{anim_class}"));
        }})();
      </script>
    "
    """
    )
    st_html(html, height=img.height + 160, scrolling=False)


# -----------------------------
# Thumbnail helpers (NEW)
# -----------------------------
def _cover(img: Image.Image, tw: int, th: int) -> Image.Image:
    """Scale & center-crop an image to fully cover (tw x th)."""
    w, h = img.size
    scale = max(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    img = img.resize((nw, nh), Image.LANCZOS)
    x0 = (nw - tw) // 2
    y0 = (nh - th) // 2
    return img.crop((x0, y0, x0 + tw, y0 + th))


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "arial.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    cx: int,
    cy: int,
    font: ImageFont.ImageFont,
    fill=(255, 255, 255),
    stroke_fill=(0, 0, 0),
    stroke_width: int = 4,
):
    bx = draw.textbbox((0, 0), text, font=font)
    w, h = bx[2] - bx[0], bx[3] - bx[1]
    draw.text(
        (cx - w // 2, cy - h // 2),
        text,
        font=font,
        fill=fill,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )


def make_thumbnail(
    panels: list[Image.Image],
    title: str | None = None,
    subtitle: str | None = None,
    layout: str = "hero",
) -> Image.Image:
    """Create a 1280x720 thumbnail from generated panels."""
    TW, TH = 1280, 720
    canvas = Image.new("RGB", (TW, TH), (10, 10, 10))
    draw = ImageDraw.Draw(canvas)

    if not panels:
        if title:
            _draw_centered_text(draw, title, TW // 2, TH // 2, _load_font(64, True))
        return canvas

    if layout == "grid" and len(panels) >= 4:
        cells = [(0, 0), (TW // 2, 0), (0, TH // 2), (TW // 2, TH // 2)]
        cw, ch = TW // 2, TH // 2
        for i in range(4):
            cell_img = _cover(panels[i], cw, ch)
            canvas.paste(cell_img, cells[i])
        # Soft dark overlay for contrast
        overlay = Image.new("RGBA", (TW, TH), (0, 0, 0, 80))
        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    else:
        idx = len(panels) // 2 if len(panels) >= 3 else 0
        hero = _cover(panels[idx], TW, TH)
        canvas.paste(hero, (0, 0))
        # Bottom gradient for legible text
        grad = Image.new("L", (1, TH), color=0)
        for y in range(TH):
            grad.putpixel((0, y), int(180 * (y / TH)))
        alpha = grad.resize((TW, TH))
        shadow = Image.new("RGBA", (TW, TH), (0, 0, 0, 0))
        shadow.putalpha(alpha)
        canvas = Image.alpha_composite(canvas.convert("RGBA"), shadow).convert("RGB")

    if title:
        title_font = _load_font(64, True)
        _draw_centered_text(
            draw,
            title.strip(),
            TW // 2,
            int(TH * 0.20),
            title_font,
            fill=(255, 255, 255),
            stroke_fill=(0, 0, 0),
            stroke_width=6,
        )
    if subtitle:
        sub_font = _load_font(38, False)
        _draw_centered_text(
            draw,
            subtitle.strip(),
            TW // 2,
            int(TH * 0.82),
            sub_font,
            fill=(255, 255, 255),
            stroke_fill=(0, 0, 0),
            stroke_width=4,
        )
    return canvas


# -----------------------------
# Content generation helpers
# -----------------------------
def b64_audio(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_dialogue(desc: str) -> str:
    prompt = (
        "Generate short, action-oriented spoken dialogue or a caption (1–2 sentences). "
        "Punchy, TV-cartoon energy. Max ~16 words.\n\n"
        f"Scene: {desc}"
    )
    resp = client.models.generate_content(model=MODEL_NAME, contents=[prompt])
    return resp.candidates[0].content.parts[0].text.strip()


def ensure_pageflip_sfx(add_page_flip: bool):
    """Load local SFX if present; else try ElevenLabs (optional)."""
    if not add_page_flip:
        return
    if st.session_state.page_flip_b64:
        return

    # 1) Local bundled asset first (no API needed)
    local = Path("assets/page_flip.mp3")
    if local.exists():
        st.session_state.page_flip_b64 = b64_audio(str(local))
        return

    # 2) Fallback to ElevenLabs SFX
    if not eleven_client:
        return
    try:
        sfx_prompt = "Quick, crisp page flip in a comic book, like turning a page"
        audio_iter = eleven_client.text_to_sound_effects.convert(
            text=sfx_prompt, duration_seconds=2, prompt_influence=0.3
        )
        save(audio_iter, "page_flip.mp3")
        st.session_state.page_flip_b64 = b64_audio("page_flip.mp3")
    except Exception as e:
        st.warning(f"Page flip SFX unavailable ({e}). Using silent flips this run.")
        st.session_state.page_flip_b64 = None


def do_generate(
    story_prompt: str,
    num_panels: int,
    style: str,
    animation_style: str,
    add_narration: bool,
    add_page_flip: bool,
):
    """Run the full generation flow and render animated panels."""
    st.session_state.panels.clear()
    st.session_state.images.clear()
    st.session_state.dialogues.clear()
    st.session_state.audio_files.clear()

    ensure_pageflip_sfx(add_page_flip)

    with st.spinner("Outlining scenes..."):
        split_prompt = (
            f"Break the following story into exactly {num_panels} concise key scenes, "
            f"numbered 1..{num_panels}:\n\n{story_prompt}"
        )
        split = client.models.generate_content(
            model=MODEL_NAME, contents=[split_prompt]
        )
        panel_text = split.candidates[0].content.parts[0].text
        st.caption("Raw scene bullets from the model:")
        st.code(panel_text)
        matches = re.findall(r"^\s*(\d+)\.\s*(.*)$", panel_text, re.MULTILINE)
        panel_descs = [
            d.strip()
            for n, d in sorted(matches, key=lambda x: int(x[0]))
            if int(n) <= num_panels
        ]
        if len(panel_descs) < num_panels:
            st.warning(
                f"Model returned {len(panel_descs)} scenes (requested {num_panels}). Proceeding with available ones."
            )

        base_prompt = (
            f"Provide a reusable description with consistent characters, outfits, settings, color palette, lighting, and motifs "
            f"for a {style} comic based on: {story_prompt}. Include details that ensure visual continuity across panels."
        )
        base_resp = client.models.generate_content(
            model=MODEL_NAME, contents=[base_prompt]
        )
        base_desc = base_resp.candidates[0].content.parts[0].text

    for i, desc in enumerate(panel_descs):
        if not desc:
            continue

        dialogue = generate_dialogue(desc)
        st.session_state.dialogues.append(dialogue)

        panel_prompt = (
            f"Panel {i+1}: {desc}\n\n"
            f"Strictly adhere to this base style bible for continuity:\n{base_desc}\n\n"
            f"Style: {style}. Dynamic comic composition, dramatic angles, square aspect ratio, "
            f"vibrant colors, textured backgrounds, high-contrast lighting.\n\n"
            f"Include a natural speech bubble with the dialogue: '{dialogue}'."
        )

        with st.spinner(f"Rendering panel {i+1}…"):
            resp = client.models.generate_content(
                model=MODEL_NAME, contents=[panel_prompt]
            )
            img: Optional[Image.Image] = None
            for part in resp.candidates[0].content.parts:
                if getattr(part, "inline_data", None):
                    img = Image.open(BytesIO(part.inline_data.data))

            if img is None:
                st.error(f"Panel {i+1}: no image returned.")
                st.session_state.images.append(Image.new("RGB", (1024, 1024), "white"))
                st.session_state.panels.append(desc)
                st.session_state.audio_files.append(None)
                continue

            st.session_state.images.append(img)
            st.session_state.panels.append(desc)

            # Animated card + (optional) flip SFX on reveal
            render_animated_panel(
                img=img,
                idx=i,
                caption=dialogue,
                sfx_b64=st.session_state.page_flip_b64 if add_page_flip else None,
                style_name=animation_style,
            )

            # Optional narration
            audio_file = None
            if add_narration and eleven_client:
                try:
                    audio_iter = eleven_client.text_to_speech.convert(
                        text=dialogue,
                        voice_id="nPczCjzI2devNBz1zQrb",
                        model_id="eleven_multilingual_v2",
                        output_format="mp3_44100_128",
                    )
                    audio_file = f"panel_{i+1}_narration.mp3"
                    save(audio_iter, audio_file)
                except Exception as e:
                    st.warning(f"Narration failed for panel {i+1}: {e}")
            st.session_state.audio_files.append(audio_file)


def do_slideshow(animation_style: str, add_page_flip: bool):
    if not st.session_state.images:
        st.info("No panels yet. Generate first.")
        return
    for i, img in enumerate(st.session_state.images):
        render_slideshow_panel(img, i, st.session_state.dialogues[i], animation_style)
        audio = st.session_state.audio_files[i]
        if audio:
            st.audio(audio, format="audio/mp3")
        time.sleep(4)
        if (
            i < len(st.session_state.images) - 1
            and add_page_flip
            and st.session_state.page_flip_b64
        ):
            st_html(
                f"""<audio autoplay style="display:none">
                       <source src="data:audio/mp3;base64,{st.session_state.page_flip_b64}" type="audio/mpeg">
                     </audio>""",
                height=0,
            )
            time.sleep(0.25)


# -----------------------------
# Buttons (manual flow)
# -----------------------------
col_g, col_s, col_v = st.columns([1, 1, 1])
with col_g:
    if st.button("Generate Comic", key="btn_generate"):
        if not story_prompt_val:
            st.warning(
                "Please enter a story prompt (or click a Quick Prompt) before generating."
            )
        else:
            do_generate(
                story_prompt_val,
                num_panels,
                style,
                animation_style,
                add_narration,
                add_page_flip,
            )

with col_s:
    if st.button("Play Comic as Slideshow", key="btn_slideshow"):
        do_slideshow(animation_style, add_page_flip)

with col_v:
    if st.button("Export to Video", key="btn_export_video"):
        with st.spinner("Rendering video…"):
            try:
                clips = []
                # optional flip SFX clip (volume trimmed)
                page_flip_audio = None
                if add_page_flip and (
                    Path("page_flip.mp3").exists() or st.session_state.page_flip_b64
                ):
                    if (
                        not Path("page_flip.mp3").exists()
                        and st.session_state.page_flip_b64
                    ):
                        with open("page_flip.mp3", "wb") as f:
                            f.write(base64.b64decode(st.session_state.page_flip_b64))
                    page_flip_audio = AudioFileClip("page_flip.mp3").fx(volumex, 0.6)

                for i, im in enumerate(st.session_state.images):
                    p = f"panel_{i+1}.png"
                    im.save(p)

                    dur = 5.0
                    narration_path = st.session_state.audio_files[i]
                    narration_clip = None
                    if narration_path and os.path.exists(narration_path):
                        narration_clip = AudioFileClip(narration_path)
                        dur = max(dur, getattr(narration_clip, "duration", 0.0) + 0.5)

                    # Build letterboxed 1080p landscape clip (prevents YouTube Shorts)
                    base = ImageClip(p).resize(height=1080)
                    base = base.on_color(
                        size=(1920, 1080), color=(0, 0, 0), pos=("center", "center")
                    ).with_duration(dur)
                    if narration_clip:
                        base = base.with_audio(narration_clip)

                    # Insert flip SFX bumper between panels
                    if i > 0 and page_flip_audio is not None:
                        flip_base = ImageClip(p).resize(height=1080)
                        flip_base = (
                            flip_base.on_color(
                                size=(1920, 1080),
                                color=(0, 0, 0),
                                pos=("center", "center"),
                            )
                            .with_duration(page_flip_audio.duration)
                            .with_audio(page_flip_audio)
                        )
                        clips.append(flip_base)

                    clips.append(base)

                video = concatenate_videoclips(clips, method="compose")
                out_path = "comic_reel.mp4"
                video.write_videofile(
                    out_path,
                    fps=30,
                    codec="libx264",
                    audio_codec="aac",
                    bitrate="6M",
                    threads=0,
                    logger=None,
                )
                # Cleanup temps
                for i in range(len(st.session_state.images)):
                    p = f"panel_{i+1}.png"
                    if os.path.exists(p):
                        os.remove(p)
                with open(out_path, "rb") as f:
                    st.download_button(
                        "Download Video Reel",
                        f,
                        file_name="comic_reel.mp4",
                        mime="video/mp4",
                        key="dl_video_reel",
                    )
                st.success("Video exported!")
            except Exception as e:
                st.error(
                    f"Video export failed: {e}. Ensure FFmpeg is available (imageio-ffmpeg wheel is bundled)."
                )

# -----------------------------
# Comic strip PNG download
# -----------------------------
if st.session_state.images:
    strip_w = max(im.width for im in st.session_state.images)
    strip_h = sum(im.height for im in st.session_state.images)
    strip = Image.new("RGB", (strip_w, strip_h), "white")
    y = 0
    for im in st.session_state.images:
        x = (strip_w - im.width) // 2
        strip.paste(im, (x, y))
        y += im.height
    strip_path = "comic_strip.png"
    strip.save(strip_path)
    with open(strip_path, "rb") as f:
        st.download_button(
            "Download Comic Strip",
            f,
            file_name="comic.png",
            mime="image/png",
            key="dl_comic_strip",
        )

# -----------------------------
# Thumbnail Generator (NEW)
# -----------------------------
with st.expander("Create YouTube Thumbnail (1280×720)"):
    default_title = "FlipBook AI"
    default_sub = (
        (story_prompt_val[:80] + "…")
        if story_prompt_val and len(story_prompt_val) > 80
        else (story_prompt_val or "")
    )
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        thumb_title = st.text_input(
            "Title (optional)", value=default_title, key="thumb_title"
        )
    with tcol2:
        thumb_sub = st.text_input(
            "Subtitle (optional)", value=default_sub, key="thumb_sub"
        )

    layout_choice = st.radio(
        "Layout",
        ["Hero (single panel)", "Grid (2×2)"],
        horizontal=True,
        key="thumb_layout",
    )
    if st.button("Generate Thumbnail", key="btn_make_thumb"):
        layout_key = "grid" if "Grid" in layout_choice else "hero"
        thumb_img = make_thumbnail(
            st.session_state.images,
            title=thumb_title,
            subtitle=thumb_sub,
            layout=layout_key,
        )
        thumb_path = "thumbnail_1280x720.png"
        thumb_img.save(thumb_path)
        st.image(thumb_img, caption="Generated Thumbnail", width="stretch")
        with open(thumb_path, "rb") as f:
            st.download_button(
                "Download Thumbnail PNG",
                f,
                file_name="thumbnail_1280x720.png",
                mime="image/png",
                key="dl_thumb_png",
            )

# -----------------------------
# DEMO MODE auto-run once
# -----------------------------
if demo_mode and not st.session_state["demo_has_run"]:
    if story_prompt_val:
        do_generate(
            story_prompt_val,
            num_panels,
            style,
            animation_style,
            add_narration,
            add_page_flip,
        )
        do_slideshow(animation_style, add_page_flip)
        st.session_state["demo_has_run"] = True
    else:
        st.info("Demo Mode is on — add a prompt or click a Quick Prompt to auto-run.")
