"""Bounded video decoding with PySceneDetect cuts and presentation timestamps."""

from bisect import bisect_left
from pathlib import Path
import math
import sys
import time

import numpy as np


def require_video():
    try:
        import cv2
        import scenedetect
    except ImportError as error:
        raise ValueError("Video dependencies are required; install .[retrieval] or .[video]") from error
    return cv2, scenedetect


def detect_shots(path, *, threshold=27.0, min_scene_frames=15):
    if not math.isfinite(threshold) or threshold <= 0 or min_scene_frames <= 0:
        raise ValueError("Shot threshold and minimum scene frames must be positive")
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Video file does not exist: {path}")
    cv2, scenedetect = require_video()
    cap = cv2.VideoCapture(str(path))
    started = time.perf_counter()
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot decode video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no valid frame rate")
        expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        detector = scenedetect.ContentDetector(threshold=threshold, min_scene_len=min_scene_frames)
        cuts, timestamps = [], []
        dimensions = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            number = len(timestamps)
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
            height, width = frame.shape[:2]
            dimensions = (width, height)
            factor = min(1.0, 320 / max(width, height))
            small = cv2.resize(frame, (max(1, round(width * factor)), max(1, round(height * factor))),
                               interpolation=cv2.INTER_AREA)
            cuts.extend(cut.frame_num for cut in detector.process_frame(
                scenedetect.FrameTimecode(number, fps=fps), small))
            if (number + 1) % 500 == 0:
                print(f"Scanned {number + 1}/{expected_frames or '?'} video frames", file=sys.stderr)
        if not timestamps:
            raise ValueError("Video contains no usable frames")
        if expected_frames > len(timestamps) + 1:
            raise ValueError(f"Video ended after {len(timestamps)} frames; header reports {expected_frames} (possibly truncated)")
        cuts.extend(cut.frame_num for cut in detector.post_process(
            scenedetect.FrameTimecode(len(timestamps) - 1, fps=fps)))
    finally:
        cap.release()
    timestamp_source = "opencv_presentation_time"
    if (not np.isfinite(timestamps).all() or timestamps[0] < 0
            or any(b <= a for a, b in zip(timestamps, timestamps[1:]))):
        timestamp_source = "frame_number_divided_by_fps"
        timestamps = [number / fps for number in range(len(timestamps))]
        print("Video timestamps unavailable/nonmonotonic; using frame-number / FPS estimates", file=sys.stderr)
    last_duration = float(np.median(np.diff(timestamps[-31:]))) if len(timestamps) > 1 else 1 / fps
    end_seconds = timestamps[-1] + last_duration
    boundaries = [0, *sorted({cut for cut in cuts if 0 < cut < len(timestamps)}), len(timestamps)]
    shots = [{"id": number + 1, "start_frame": start, "end_frame": end,
              "start_seconds": timestamps[start],
              "end_seconds": timestamps[end] if end < len(timestamps) else end_seconds}
             for number, (start, end) in enumerate(zip(boundaries, boundaries[1:]))]
    return {"path": str(path), "fps": fps, "decoded_frames": len(timestamps),
            "width": dimensions[0], "height": dimensions[1], "end_seconds": end_seconds,
            "timestamp_source": timestamp_source, "final_frame_duration_estimated": True,
            "scene_detector": {"name": "ContentDetector", "threshold": threshold,
                               "min_scene_frames": min_scene_frames, "max_side": 320},
            "versions": {"opencv": cv2.__version__, "scenedetect": scenedetect.__version__},
            "scan_seconds": time.perf_counter() - started}, shots, timestamps


def select_frames(shot: dict, timestamps: list[float], interval_seconds=None) -> list[int]:
    if interval_seconds is not None and (not math.isfinite(interval_seconds) or interval_seconds <= 0):
        raise ValueError("interval_seconds must be positive and finite")
    start, end = shot["start_frame"], shot["end_frame"]
    targets = [(shot["start_seconds"] + shot["end_seconds"]) / 2]
    if interval_seconds is not None:
        duration = shot["end_seconds"] - shot["start_seconds"]
        spacing = max(interval_seconds, duration / max(1, end - start))
        targets.extend(shot["start_seconds"] + step * spacing
                       for step in range(1, math.ceil(duration / spacing)))
    frames = set()
    for target in targets:
        insertion = min(end - 1, bisect_left(timestamps, target, lo=start, hi=end))
        candidates = [insertion]
        if insertion > start:
            candidates.append(insertion - 1)
        frames.add(min(candidates, key=lambda number: (abs(timestamps[number] - target), number)))
    return sorted(frames)


def read_selected_frames(path, frame_numbers):
    """Advance sequentially and retrieve selected frames, avoiding inexact seeks."""
    cv2, _ = require_video()
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot reopen video: {path}")
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        current = -1
        for number in frame_numbers:
            if number <= current:
                raise ValueError("Selected frames must be unique and increasing")
            while current < number:
                if not cap.grab():
                    raise ValueError(f"Video ended before selected frame {number}")
                current += 1
            ok, frame = cap.retrieve()
            actual = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) - 1
            if not ok or actual != number:
                raise ValueError(f"Cannot decode exact frame {number}; decoder reported {actual}")
            yield number, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
