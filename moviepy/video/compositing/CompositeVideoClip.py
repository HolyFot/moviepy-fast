"""Main video composition interface of MoviePy."""

from functools import reduce

import cv2
import numpy as np
from PIL import Image

from moviepy.audio.AudioClip import CompositeAudioClip
from moviepy.video.VideoClip import ColorClip, VideoClip


class CompositeVideoClip(VideoClip):
    """
    A VideoClip made of other videoclips displayed together. This is the
    base class for most compositions.

    Parameters
    ----------

    size
      The size (width, height) of the final clip.

    clips
      A list of videoclips.

      Clips with a higher ``layer`` attribute will be displayed
      on top of other clips in a lower layer.
      If two or more clips share the same ``layer``,
      then the one appearing latest in ``clips`` will be displayed
      on top (i.e. it has the higher layer).

      For each clip:

      - The attribute ``pos`` determines where the clip is placed.
          See ``VideoClip.set_pos``
      - The mask of the clip determines which parts are visible.

      Finally, if all the clips in the list have their ``duration``
      attribute set, then the duration of the composite video clip
      is computed automatically

    bg_color
      Color for the unmasked and unfilled regions. Set to None for these
      regions to be transparent (will be slower).
      Default is black (0, 0, 0).

    use_bgclip
      Set to True if the first clip in the list should be used as the
      'background' on which all other clips are blitted. That first clip must
      have the same size as the final clip. If it has no transparency, the final
      clip will have no mask.

    The clip with the highest FPS will be the FPS of the composite clip.

    """

    def __init__(
        self, clips, size=None, bg_color=None, use_bgclip=False, is_mask=False
    ):
        if size is None:
            size = clips[0].size

        if use_bgclip and (clips[0].mask is None):
            transparent = False
        else:
            transparent = True if bg_color is None else False

        # If we must not use first clip as background and we dont have a color
        # we generate a black background if clip should not be transparent and
        # a transparent background if transparent
        if (not use_bgclip) and bg_color is None:
            if transparent:
                bg_color = 0.0 if is_mask else (0, 0, 0, 0)
            else:
                bg_color = 0.0 if is_mask else (0, 0, 0)

        fpss = [clip.fps for clip in clips if getattr(clip, "fps", None)]
        self.fps = max(fpss) if fpss else None

        VideoClip.__init__(self)

        self.size = size
        self.is_mask = is_mask
        self.clips = clips
        self.bg_color = bg_color

        # Use first clip as background if necessary, else use color
        # either set by user or previously generated
        if use_bgclip:
            self.bg = clips[0]
            self.clips = clips[1:]
            self.created_bg = False
        else:
            self.clips = clips
            self.bg = ColorClip(size, color=self.bg_color, is_mask=is_mask)
            self.created_bg = True

        # order self.clips by layer
        self.clips = sorted(self.clips, key=lambda clip: clip.layer_index)

        # compute duration
        ends = [clip.end for clip in self.clips]
        if None not in ends:
            duration = max(ends)
            self.duration = duration
            self.end = duration

        # compute audio
        audioclips = [v.audio for v in self.clips if v.audio is not None]
        if audioclips:
            self.audio = CompositeAudioClip(audioclips)

        # compute mask if necessary
        if transparent:
            maskclips = [
                (clip.mask if (clip.mask is not None) else clip.with_mask().mask)
                .with_position(clip.pos)
                .with_end(clip.end)
                .with_start(clip.start, change_end=False)
                .with_layer_index(clip.layer_index)
                for clip in self.clips
            ]

            if use_bgclip and self.bg.mask:
                maskclips = [self.bg.mask] + maskclips

            self.mask = CompositeVideoClip(
                maskclips, self.size, is_mask=True, bg_color=0.0
            )

        # Pre-compute compositing optimizations
        self._cached_frame = None
        self._clip_blit_pil = {}
        self._clip_blit_np = {}
        self._any_masks = any(c.mask is not None for c in self.clips)
        if not is_mask:
            self._precompute_compositing()

    def _precompute_compositing(self):
        """Pre-compute blit data for static clips and cache fully static frames."""
        full_w, full_h = self.size
        all_static = True
        all_always_playing = True

        for i, clip in enumerate(self.clips):
            src_pil = getattr(clip, '_cached_pil', None)
            has_mask = clip.mask is not None
            mask_pil = (getattr(clip.mask, '_cached_pil', None)
                        if has_mask else None)

            if src_pil is None or (has_mask and mask_pil is None):
                all_static = False
                continue

            # Check constant position
            if clip.pos(0) != clip.pos(1.0):
                all_static = False
                continue

            # Resolve position via new_blit_on (handles string positions)
            result = clip.new_blit_on(clip.start, full_w, full_h)
            if result is None:
                continue
            _, pos, _, _ = result
            x, y = pos
            cw_px, ch_px = src_pil.size  # PIL size is (w, h)

            if x <= -cw_px or x >= full_w or y <= -ch_px or y >= full_h:
                continue

            src_x1, src_y1 = max(0, -x), max(0, -y)
            src_x2 = min(cw_px, full_w - x)
            src_y2 = min(ch_px, full_h - y)
            dst_x1, dst_y1 = max(0, x), max(0, y)

            if src_x2 <= src_x1 or src_y2 <= src_y1:
                continue

            no_crop = (src_x1 == 0 and src_y1 == 0
                       and src_x2 == cw_px and src_y2 == ch_px)

            # Pre-compute PIL blit data
            sp = src_pil if no_crop else src_pil.crop((src_x1, src_y1, src_x2, src_y2))
            mp = None
            if mask_pil is not None:
                mp = mask_pil if no_crop else mask_pil.crop(
                    (src_x1, src_y1, src_x2, src_y2))
            self._clip_blit_pil[i] = (sp, mp, dst_x1, dst_y1, has_mask)

            # Pre-compute numpy blit data
            cached = clip._cached_uint8
            if cached is not None:
                dst_x2 = dst_x1 + (src_x2 - src_x1)
                dst_y2 = dst_y1 + (src_y2 - src_y1)
                src_slice = cached[src_y1:src_y2, src_x1:src_x2]
                if src_slice.ndim == 2:
                    src_slice = cv2.cvtColor(src_slice, cv2.COLOR_GRAY2RGB)
                elif src_slice.ndim == 3 and src_slice.shape[2] == 4:
                    src_slice = src_slice[:, :, :3]
                self._clip_blit_np[i] = (src_slice, dst_y1, dst_y2, dst_x1, dst_x2)

            # Check always playing
            if clip.start != 0:
                all_always_playing = False
            elif (self.duration is not None and clip.end is not None
                  and clip.end < self.duration):
                all_always_playing = False

        # Fully static + always playing → pre-render one frame
        if all_static and all_always_playing and self._clip_blit_pil:
            bg_pil = getattr(self.bg, '_cached_pil', None)
            if bg_pil is not None and bg_pil.size == (full_w, full_h):
                canvas = bg_pil.copy()
            else:
                canvas = Image.fromarray(
                    self._make_numpy_canvas(0, full_w, full_h), mode='RGB'
                )
            for i in sorted(self._clip_blit_pil):
                sp, mp, dx, dy, has_mask = self._clip_blit_pil[i]
                if has_mask and mp is not None:
                    canvas.paste(sp, (dx, dy), mp)
                else:
                    canvas.paste(sp, (dx, dy))
            self._cached_frame = np.asarray(canvas).copy()

    def frame_function(self, t):
        """The clips playing at time `t` are blitted over one another."""
        full_w, full_h = self.size

        # Mask compositing path — pure numpy
        if self.is_mask:
            mask = np.zeros((full_h, full_w), dtype=np.float32)
            for clip in self.playing_clips(t):
                mask = clip.compose_mask(mask, t)
            return mask

        # Fully static composition → return pre-rendered frame
        if self._cached_frame is not None:
            return self._cached_frame

        if self._any_masks:
            return self._frame_pil_canvas(t, full_w, full_h)
        else:
            return self._frame_numpy(t, full_w, full_h)

    def _make_numpy_canvas(self, t, full_w, full_h):
        """Build a writable uint8 RGB numpy canvas from the background."""
        bg = self.bg.get_frame_uint8(t - self.bg.start)
        if bg.ndim == 2:
            bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2RGB)
        elif bg.shape[2] == 4:
            bg = bg[:, :, :3]
        if bg.shape[0] >= full_h and bg.shape[1] >= full_w:
            return bg[:full_h, :full_w].copy()
        current = np.zeros((full_h, full_w, 3), dtype=np.uint8)
        bh, bw = bg.shape[:2]
        current[:min(bh, full_h), :min(bw, full_w)] = \
            bg[:min(bh, full_h), :min(bw, full_w)]
        return current

    def _frame_pil_canvas(self, t, full_w, full_h):
        """Compositing via a single PIL canvas with pre-computed blit data."""
        bg_pil = getattr(self.bg, '_cached_pil', None)
        if bg_pil is not None and bg_pil.size == (full_w, full_h):
            canvas = bg_pil.copy()
        else:
            canvas = Image.fromarray(
                self._make_numpy_canvas(t, full_w, full_h), mode='RGB'
            )

        blit_cache = self._clip_blit_pil

        for i, clip in enumerate(self.clips):
            # Inline is_playing — skip decorator overhead
            if not (clip.start <= t and (clip.end is None or t < clip.end)):
                continue

            blit = blit_cache.get(i)
            if blit is not None:
                # Fast path: pre-computed PIL data
                sp, mp, dx, dy, has_mask = blit
                if has_mask and mp is not None:
                    canvas.paste(sp, (dx, dy), mp)
                else:
                    canvas.paste(sp, (dx, dy))
            else:
                # Fallback for dynamic clips
                result = clip.new_blit_on(t, full_w, full_h)
                if result is None:
                    continue
                img, pos, clip_mask, _ = result
                if img is None:
                    continue

                clip_h, clip_w = img.shape[:2]
                x, y = pos
                if x <= -clip_w or x >= full_w or y <= -clip_h or y >= full_h:
                    continue

                src_x1, src_y1 = max(0, -x), max(0, -y)
                src_x2 = min(clip_w, full_w - x)
                src_y2 = min(clip_h, full_h - y)
                dst_x1, dst_y1 = max(0, x), max(0, y)

                if src_x2 <= src_x1 or src_y2 <= src_y1:
                    continue

                src_arr = img[src_y1:src_y2, src_x1:src_x2]
                if src_arr.ndim == 2:
                    src_arr = cv2.cvtColor(src_arr, cv2.COLOR_GRAY2RGB)
                elif src_arr.shape[2] == 4:
                    src_arr = src_arr[:, :, :3]

                sp = Image.fromarray(src_arr, mode='RGB')
                if clip_mask is not None:
                    mask_s = clip_mask[src_y1:src_y2, src_x1:src_x2]
                    if mask_s.ndim == 3:
                        mask_s = mask_s[:, :, 0]
                    mp = Image.fromarray(mask_s, mode='L')
                    canvas.paste(sp, (dst_x1, dst_y1), mp)
                else:
                    canvas.paste(sp, (dst_x1, dst_y1))

        return np.asarray(canvas)

    def _frame_numpy(self, t, full_w, full_h):
        """Pure numpy compositing — fastest for clips without masks."""
        current = self._make_numpy_canvas(t, full_w, full_h)

        np_cache = self._clip_blit_np

        for i, clip in enumerate(self.clips):
            # Inline is_playing — skip decorator overhead
            if not (clip.start <= t and (clip.end is None or t < clip.end)):
                continue

            np_data = np_cache.get(i)
            if np_data is not None:
                # Fast path: pre-computed numpy slices
                src, dy1, dy2, dx1, dx2 = np_data
                current[dy1:dy2, dx1:dx2] = src
            else:
                # Fallback for dynamic clips
                result = clip.new_blit_on(t, full_w, full_h)
                if result is None:
                    continue
                img, pos, clip_mask, _ = result
                if img is None:
                    continue

                clip_h, clip_w = img.shape[:2]
                x, y = pos
                if x <= -clip_w or x >= full_w or y <= -clip_h or y >= full_h:
                    continue

                src_x1, src_y1 = max(0, -x), max(0, -y)
                src_x2 = min(clip_w, full_w - x)
                src_y2 = min(clip_h, full_h - y)
                dst_x1, dst_y1 = max(0, x), max(0, y)
                dst_x2 = dst_x1 + (src_x2 - src_x1)
                dst_y2 = dst_y1 + (src_y2 - src_y1)

                if src_x2 <= src_x1 or src_y2 <= src_y1:
                    continue

                src = img[src_y1:src_y2, src_x1:src_x2]
                if src.ndim == 2:
                    src = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
                elif src.shape[2] == 4:
                    src = src[:, :, :3]
                current[dst_y1:dst_y2, dst_x1:dst_x2] = src

        return current

    def playing_clips(self, t=0):
        """Returns a list of the clips in the composite clips that are
        actually playing at the given time `t`.
        """
        return [clip for clip in self.clips if clip.is_playing(t)]

    def close(self):
        """Closes the instance, releasing all the resources."""
        if self.created_bg and self.bg:
            # Only close the background clip if it was locally created.
            # Otherwise, it remains the job of whoever created it.
            self.bg.close()
            self.bg = None
        if hasattr(self, "audio") and self.audio:
            self.audio.close()
            self.audio = None


def clips_array(array, rows_widths=None, cols_heights=None, bg_color=None):
    """Given a matrix whose rows are clips, creates a CompositeVideoClip where
    all clips are placed side by side horizontally for each clip in each row
    and one row on top of the other for each row. So given next matrix of clips
    with same size:

    ```python
    clips_array([[clip1, clip2, clip3], [clip4, clip5, clip6]])
    ```

    the result will be a CompositeVideoClip with a layout displayed like:

    ```
    ┏━━━━━━━┳━━━━━━━┳━━━━━━━┓
    ┃       ┃       ┃       ┃
    ┃ clip1 ┃ clip2 ┃ clip3 ┃
    ┃       ┃       ┃       ┃
    ┣━━━━━━━╋━━━━━━━╋━━━━━━━┫
    ┃       ┃       ┃       ┃
    ┃ clip4 ┃ clip5 ┃ clip6 ┃
    ┃       ┃       ┃       ┃
    ┗━━━━━━━┻━━━━━━━┻━━━━━━━┛
    ```

    If some clips doesn't fulfill the space required by the rows or columns
    in which are placed, that space will be filled by the color defined in
    ``bg_color``.

    array
      Matrix of clips included in the returned composited video clip.

    rows_widths
      Widths of the different rows in pixels. If ``None``, is set automatically.

    cols_heights
      Heights of the different columns in pixels. If ``None``, is set automatically.

    bg_color
       Fill color for the masked and unfilled regions. Set to ``None`` for these
       regions to be transparent (processing will be slower).
    """
    array = np.array(array)
    sizes_array = np.array([[clip.size for clip in line] for line in array])

    # find row width and col_widths automatically if not provided
    if rows_widths is None:
        rows_widths = sizes_array[:, :, 1].max(axis=1)
    if cols_heights is None:
        cols_heights = sizes_array[:, :, 0].max(axis=0)

    # compute start positions of X for rows and Y for columns
    xs = np.cumsum([0] + list(cols_heights))
    ys = np.cumsum([0] + list(rows_widths))

    for j, (x, ch) in enumerate(zip(xs[:-1], cols_heights)):
        for i, (y, rw) in enumerate(zip(ys[:-1], rows_widths)):
            clip = array[i, j]
            w, h = clip.size
            # if clip not fulfill row width or column height
            if (w < ch) or (h < rw):
                clip = CompositeVideoClip(
                    [clip.with_position("center")], size=(ch, rw), bg_color=bg_color
                ).with_duration(clip.duration)

            array[i, j] = clip.with_position((x, y))

    return CompositeVideoClip(array.flatten(), size=(xs[-1], ys[-1]), bg_color=bg_color)


def concatenate_videoclips(
    clips, method="chain", transition=None, bg_color=None, is_mask=False, padding=0
):
    """Concatenates several video clips.

    Returns a video clip made by clip by concatenating several video clips.
    (Concatenated means that they will be played one after another).

    There are two methods:

    - method="chain": will produce a clip that simply outputs
      the frames of the successive clips, without any correction if they are
      not of the same size of anything. If none of the clips have masks the
      resulting clip has no mask, else the mask is a concatenation of masks
      (using completely opaque for clips that don't have masks, obviously).
      If you have clips of different size and you want to write directly the
      result of the concatenation to a file, use the method "compose" instead.

    - method="compose", if the clips do not have the same resolution, the final
      resolution will be such that no clip has to be resized.
      As a consequence the final clip has the height of the highest clip and the
      width of the widest clip of the list. All the clips with smaller dimensions
      will appear centered. The border will be transparent if mask=True, else it
      will be of the color specified by ``bg_color``.

    The clip with the highest FPS will be the FPS of the result clip.

    Parameters
    ----------
    clips
      A list of video clips which must all have their ``duration``
      attributes set.
    method
      "chain" or "compose": see above.
    transition
      A clip that will be played between each two clips of the list.

    bg_color
      Only for method='compose'. Color of the background.
      Set to None for a transparent clip

    padding
      Only for method='compose'. Duration during two consecutive clips.
      Note that for negative padding, a clip will partly play at the same
      time as the clip it follows (negative padding is cool for clips who fade
      in on one another). A non-null padding automatically sets the method to
      `compose`.

    """
    if transition is not None:
        clip_transition_pairs = [[v, transition] for v in clips[:-1]]
        clips = reduce(lambda x, y: x + y, clip_transition_pairs) + [clips[-1]]
        transition = None

    timings = np.cumsum([0] + [clip.duration for clip in clips])

    sizes = [clip.size for clip in clips]

    w = max(size[0] for size in sizes)
    h = max(size[1] for size in sizes)

    timings = np.maximum(0, timings + padding * np.arange(len(timings)))
    timings[-1] -= padding  # Last element is the duration of the whole

    if method == "chain":

        def frame_function(t):
            i = max([i for i, e in enumerate(timings) if e <= t])
            return clips[i].get_frame(t - timings[i])

        def get_mask(clip):
            mask = clip.mask or ColorClip(clip.size, color=1, is_mask=True)
            if mask.duration is None:
                mask.duration = clip.duration
            return mask

        result = VideoClip(is_mask=is_mask, frame_function=frame_function)
        if any([clip.mask is not None for clip in clips]):
            masks = [get_mask(clip) for clip in clips]
            result.mask = concatenate_videoclips(masks, method="chain", is_mask=True)
            result.clips = clips
    elif method == "compose":
        result = CompositeVideoClip(
            [
                clip.with_start(t).with_position("center")
                for (clip, t) in zip(clips, timings)
            ],
            size=(w, h),
            bg_color=bg_color,
            is_mask=is_mask,
        )
    else:
        raise Exception(
            "MoviePy Error: The 'method' argument of "
            "concatenate_videoclips must be 'chain' or 'compose'"
        )

    result.timings = timings

    result.start_times = timings[:-1]
    result.start, result.duration, result.end = 0, timings[-1], timings[-1]

    audio_t = [
        (clip.audio, t) for clip, t in zip(clips, timings) if clip.audio is not None
    ]
    if audio_t:
        result.audio = CompositeAudioClip(
            [a.with_start(t) for a, t in audio_t]
        ).with_duration(result.duration)

    fpss = [clip.fps for clip in clips if getattr(clip, "fps", None) is not None]
    result.fps = max(fpss) if fpss else None
    return result
