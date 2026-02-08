"""Example: karaoke-style highlighted subtitles with FancyText.

This script creates a showcase video that demonstrates the full range of
FancyText capabilities:

 1. **Basic karaoke** – word-by-word highlight with shadow + highlight box.
 2. **Golden gradient** – gradient overlay + bevel & emboss for a premium look.
 3. **Chrome / Metallic** – silver-chrome preset for cinematic text.
 4. **Inner glow + inner shadow** – soft inner effects for a subtle style.
 5. **Neon glow** – outer glow (no shadow) with bright colours.

Usage
-----
    python fancytext_subtitles.py

The output is written to ``fancytext_subtitles_demo.mp4`` in the current
directory.
"""

import sys
import os

# Ensure the local moviepy package is used instead of any installed version
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from moviepy import (
    ColorClip,
    CompositeVideoClip,
    create_word_fancytext,
    create_word_fancytext_adv,
    create_static_fancytext,
    create_text_overlay,
    generate_procedural_texture,
)

# ── Video settings ──────────────────────────────────────────────
WIDTH, HEIGHT = 1280, 720
FPS = 24
TOTAL_DURATION = 55.0  # seconds

# ── Section 1: Basic karaoke subtitles ──────────────────────────
BASIC_SUBS = [
    {"text": "Welcome to the MoviePy demo", "start": 0.5, "duration": 3.0},
    {"text": "Basic karaoke-style word highlight", "start": 4.0, "duration": 3.5},
]

# ── Section 2: Golden gradient + bevel/emboss ──────────────────
GRADIENT_SUBS = [
    {"text": "Now with a golden gradient overlay", "start": 8.5, "duration": 3.5},
    {"text": "Plus bevel and emboss for depth", "start": 12.5, "duration": 3.5},
]

# ── Section 3: Chrome / Metallic ──────────────────────────────
CHROME_SUBS = [
    {"text": "Check out this chrome effect", "start": 17.0, "duration": 3.5},
    {"text": "Silver metallic text looks cinematic", "start": 21.0, "duration": 3.5},
]

# ── Section 4: Inner glow + inner shadow ──────────────────────
INNER_FX_SUBS = [
    {"text": "Inner glow gives a soft feel", "start": 25.5, "duration": 3.5},
    {"text": "Combined with inner shadow for depth", "start": 29.5, "duration": 3.5},
]

# ── Section 5: Neon glow (no shadow) ──────────────────────────
NEON_SUBS = [
    {"text": "Neon glow style for vibrant text", "start": 34.0, "duration": 3.5},
    {"text": "Great for music videos and intros", "start": 38.0, "duration": 3.5},
]

# ── Section 6: Lava texture ───────────────────────────────────
TEXTURE_SUBS = [
    {"text": "Lava texture with molten glow", "start": 42.5, "duration": 3.5},
    {"text": "Procedural lava with bevel", "start": 46.5, "duration": 3.5},
]


def _shift_clips(clips, global_start):
    """Shift clip start times by *global_start* and sync masks."""
    shifted = []
    for clip in clips:
        c = clip.with_start(clip.start + global_start)
        if c.mask is not None:
            c.mask = c.mask.with_start(c.start)
        shifted.append(c)
    return shifted


def _build_basic_karaoke(subs):
    """Section 1 – plain word-by-word highlight with shadow + box."""
    clips = []
    for sub in subs:
        word_clips = create_word_fancytext(
            text=sub["text"],
            duration=sub["duration"],
            width=WIDTH,
            height=HEIGHT,
            font_size=65,
            text_color=(255, 255, 255),
            highlight_color=(255, 255, 0),
            stroke_color=(0, 0, 0),
            stroke_width=4,
            max_words_per_line=6,
            position_y_ratio=0.72,
            shadow_enabled=True,
            shadow_offset=(2, 2),
            shadow_blur=4,
            glow_enabled=False,
            highlight_box_enabled=True,
            highlight_box_color=(0, 80, 200),
            highlight_box_radius=10,
            highlight_box_blur=8,
            highlight_box_opacity=0.3,
        )
        clips.extend(_shift_clips(word_clips, sub["start"]))
    return clips


def _build_gradient_bevel(subs):
    """Section 2 – gradient overlay + bevel & emboss."""
    clips = []
    for sub in subs:
        line_clips = create_word_fancytext_adv(
            text=sub["text"],
            duration=sub["duration"],
            width=WIDTH,
            height=HEIGHT,
            font_size=65,
            text_color=(255, 255, 255),
            highlight_color=(255, 220, 80),        # warm gold highlight
            stroke_color=(80, 50, 0),              # dark gold outline
            stroke_width=3,
            max_words_per_line=6,
            position_y_ratio=0.72,
            shadow_enabled=True,
            shadow_offset=(3, 3),
            shadow_blur=5,
            highlight_box_enabled=True,
            highlight_box_color=(180, 140, 0),     # gold-tinted box
            highlight_box_radius=12,
            highlight_box_blur=10,
            highlight_box_opacity=0.2,
            # ── Gradient overlay ──
            gradient_overlay_enabled=True,
            gradient_overlay_colors=((255, 230, 60), (180, 100, 0), (255, 200, 30)),
            gradient_overlay_angle=90,             # top-to-bottom
            gradient_overlay_opacity=0.85,
            # ── Bevel & Emboss (subtle) ──
            bevel_emboss_enabled=True,
            bevel_emboss_style="inner_bevel",
            bevel_emboss_depth=5,
            bevel_emboss_angle=135,
            bevel_emboss_highlight_color=(255, 255, 200),
            bevel_emboss_shadow_color=(50, 20, 0),
            bevel_emboss_highlight_opacity=0.8,
            bevel_emboss_shadow_opacity=0.65,
            bevel_emboss_soften=3,
        )
        clips.extend(_shift_clips(line_clips, sub["start"]))
    return clips


def _build_chrome(subs):
    """Section 3 – chrome / metallic preset."""
    clips = []
    for sub in subs:
        line_clips = create_word_fancytext_adv(
            text=sub["text"],
            duration=sub["duration"],
            width=WIDTH,
            height=HEIGHT,
            font_size=65,
            text_color=(200, 200, 210),
            highlight_color=(240, 240, 255),
            stroke_color=(15, 15, 25),             # darker stroke for chrome contrast
            stroke_width=3,
            max_words_per_line=6,
            position_y_ratio=0.72,
            shadow_enabled=True,
            shadow_offset=(2, 4),
            shadow_blur=6,
            highlight_box_enabled=False,
            # ── Chrome (applies its own gradient + bevel internally) ──
            chrome_enabled=True,
            chrome_colors=None,                    # classic silver bands
            chrome_opacity=0.85,
        )
        clips.extend(_shift_clips(line_clips, sub["start"]))
    return clips


def _build_inner_fx(subs):
    """Section 4 – inner glow + inner shadow."""
    clips = []
    for sub in subs:
        line_clips = create_word_fancytext_adv(
            text=sub["text"],
            duration=sub["duration"],
            width=WIDTH,
            height=HEIGHT,
            font_size=65,
            text_color=(200, 220, 255),            # pale blue
            highlight_color=(100, 200, 255),        # brighter blue highlight
            stroke_color=(0, 0, 60),
            stroke_width=3,
            max_words_per_line=6,
            position_y_ratio=0.72,
            shadow_enabled=True,
            shadow_offset=(2, 2),
            shadow_blur=4,
            highlight_box_enabled=True,
            highlight_box_color=(0, 40, 120),
            highlight_box_radius=10,
            highlight_box_blur=8,
            highlight_box_opacity=0.25,
            # ── Inner Shadow ──
            inner_shadow_enabled=True,
            inner_shadow_color=(0, 0, 40),
            inner_shadow_offset=(3, 3),
            inner_shadow_blur=5,
            inner_shadow_opacity=0.8,
            # ── Inner Glow ──
            inner_glow_enabled=True,
            inner_glow_color=(180, 220, 255),
            inner_glow_size=6,
            inner_glow_opacity=0.65,
        )
        clips.extend(_shift_clips(line_clips, sub["start"]))
    return clips


def _build_neon_glow(subs):
    """Section 5 – neon glow (shadow off, glow on)."""
    clips = []
    for sub in subs:
        word_clips = create_word_fancytext(
            text=sub["text"],
            duration=sub["duration"],
            width=WIDTH,
            height=HEIGHT,
            font_size=65,
            text_color=(255, 255, 255),
            highlight_color=(0, 255, 180),          # electric green highlight
            stroke_color=(0, 60, 40),
            stroke_width=3,
            max_words_per_line=6,
            position_y_ratio=0.72,
            # Shadow OFF so glow takes effect
            shadow_enabled=False,
            # ── Glow ──
            glow_enabled=True,
            glow_color=(0, 255, 160),
            glow_size=8,
            highlight_box_enabled=False,
        )
        clips.extend(_shift_clips(word_clips, sub["start"]))
    return clips


def _build_rock_texture(subs):
    """Section 6 – lava / molten rock texture with bevel."""
    # Procedural lava texture: black matrix with red-orange veins
    rock_tex = generate_procedural_texture(
        width=256,
        height=256,
        base_color=(5, 0, 0),            # near-black
        accent_color=(255, 60, 10),      # bright orange-red lava
        roughness=14,
        seed=77,
    )
    clips = []
    for sub in subs:
        line_clips = create_word_fancytext_adv(
            text=sub["text"],
            duration=sub["duration"],
            width=WIDTH,
            height=HEIGHT,
            font_size=65,
            text_color=(100, 30, 10),              # dark base
            highlight_color=(255, 100, 30),        # bright lava highlight
            stroke_color=(20, 5, 0),
            stroke_width=2,                        # thin stroke gives bevel more edge area
            max_words_per_line=6,
            position_y_ratio=0.72,
            shadow_enabled=False,
            glow_enabled=True,
            glow_color=(255, 80, 20),
            glow_size=10,
            highlight_box_enabled=False,
            # ── Texture ──
            texture_enabled=True,
            texture_image=rock_tex,
            texture_scale=1.0,
            texture_opacity=0.85,
            texture_blend_mode="normal",
            # ── Bevel for 3-D stone look ──
            bevel_emboss_enabled=True,
            bevel_emboss_style="inner_bevel",
            bevel_emboss_depth=7,
            bevel_emboss_angle=120,
            bevel_emboss_highlight_color=(255, 200, 120),
            bevel_emboss_shadow_color=(0, 0, 0),
            bevel_emboss_highlight_opacity=0.85,
            bevel_emboss_shadow_opacity=0.9,
            bevel_emboss_soften=4,
        )
        clips.extend(_shift_clips(line_clips, sub["start"]))
    return clips


def main():
    # ── Background ──────────────────────────────────────────────
    background = ColorClip(
        size=(WIDTH, HEIGHT),
        color=(30, 30, 45),                        # dark blue-grey
    ).with_duration(TOTAL_DURATION)

    # ── Persistent title at the top ────────────────────────────
    title = create_text_overlay(
        text="FancyText Subtitle Demo",
        duration=TOTAL_DURATION,
        width=WIDTH,
        height=HEIGHT,
        font_size=50,
        text_color=(255, 220, 100),
        stroke_color=(0, 0, 0),
        stroke_width=3,
        position="top",
    )

    # ── Section labels (centred, appear at the start of each section) ──
    section_labels = []
    label_specs = [
        ("1: Basic Karaoke", 0.0, 7.5),
        ("2: Golden Gradient + Bevel", 8.0, 8.5),
        ("3: Chrome / Metallic", 16.5, 8.0),
        ("4: Inner Glow + Shadow", 25.0, 8.0),
        ("5: Neon Glow", 33.5, 8.0),
        ("6: Lava Texture", 42.0, 9.0),
    ]
    for text, start, dur in label_specs:
        lbl = create_text_overlay(
            text=text,
            duration=dur,
            width=WIDTH,
            height=HEIGHT,
            font_size=36,
            text_color=(180, 180, 200),
            stroke_color=(0, 0, 0),
            stroke_width=2,
            position="center",
        ).with_start(start)
        if lbl.mask is not None:
            lbl.mask = lbl.mask.with_start(lbl.start)
        section_labels.append(lbl)

    # ── Build all subtitle sections ────────────────────────────
    all_clips = []
    all_clips.extend(_build_basic_karaoke(BASIC_SUBS))
    all_clips.extend(_build_gradient_bevel(GRADIENT_SUBS))
    all_clips.extend(_build_chrome(CHROME_SUBS))
    all_clips.extend(_build_inner_fx(INNER_FX_SUBS))
    all_clips.extend(_build_neon_glow(NEON_SUBS))
    all_clips.extend(_build_rock_texture(TEXTURE_SUBS))

    # ── Outro static subtitle ──────────────────────────────────
    outro = create_static_fancytext(
        text="Thanks for watching!",
        duration=2.5,
        width=WIDTH,
        height=HEIGHT,
        font_size=55,
        text_color=(255, 255, 255),
        stroke_color=(0, 0, 0),
        stroke_width=3,
        bg_opacity=0.6,
        position_y_ratio=0.45,
    ).with_start(TOTAL_DURATION - 3.0)

    if outro.mask is not None:
        outro.mask = outro.mask.with_start(outro.start)

    # ── Composite and render ───────────────────────────────────
    final = CompositeVideoClip(
        [background, title, *section_labels, *all_clips, outro],
        size=(WIDTH, HEIGHT),
    ).with_duration(TOTAL_DURATION).with_fps(FPS)

    output_path = "fancytext_subtitles_demo.mp4"
    final.write_videofile(
        output_path, fps=FPS, codec="libx264", audio=False,
        preset="ultrafast",
        threads=4,
    )
    print(f"\nDone! Output saved to: {output_path}")


if __name__ == "__main__":
    main()
