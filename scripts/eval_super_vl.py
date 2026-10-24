"""Score the Nemotron 3.5 Super VL review on the 39 episodes Assay flagged in 5 public LeRobot datasets.

For each flagged episode with local video: rebuild the verdict exactly as `assay check --judges robometer,topreward`
does, ask the rule-generated question once with the clip as a video (base64 mp4) and once with 8 keyframes, and
compare the answer with the human review in configs/public_review_labels.json. Answers are cached per episode,
input and prompt version under cache/super_vl/, so a second run costs nothing. Writes
results/super_vl_review.json. The key is read from NEBIUS_API_KEY and never printed or saved.

    uv run python scripts/eval_super_vl.py --dry-run   # build every request, no network, no key
    uv run python scripts/eval_super_vl.py             # needs NEBIUS_API_KEY in this terminal
"""

from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import av
import numpy as np

from assay.diagnose import diagnose_batch
from assay.ingest import load
from assay.judges.cache import load_curves
from assay.keyframes import jpeg_data_uri
from assay.review import MODEL, PROMPT_VERSION, review, review_requests
from assay.summary import BASE_URL, token_factory_body, token_factory_text
from assay.types import VideoRef
from assay.video import frames_at, read_clip

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = ROOT / "cache"
LABELS = json.loads((ROOT / "configs" / "public_review_labels.json").read_text(encoding="utf-8"))["labels"]
CAMERA = {"CoRL2026-CSI__SO101-DualArm-teleop_fold_towel_100epi": "observation.images.left_top"}  # the camera the judges used
VIDEO_FPS = 10  # the clip is re-encoded at this rate; the model samples about 2 frames per second anyway
MAX_SIDE = 448
N_FRAMES = 8


def jpeg_bytes(img: np.ndarray) -> bytes:
    return base64.b64decode(jpeg_data_uri(img).split(",", 1)[1])


def mp4_bytes(frames: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with av.open(buf, "w", format="mp4") as out:
        s = out.add_stream("libx264", rate=VIDEO_FPS)
        s.height, s.width = frames.shape[1:3]
        s.pix_fmt = "yuv420p"
        for img in frames:
            for p in s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")):
                out.mux(p)
        for p in s.encode(None):
            out.mux(p)
    return buf.getvalue()


def media(ref: VideoRef, t0: float, t1: float) -> tuple[bytes, list[bytes], list[float]]:
    """The clip [t0, t1] (seconds from the episode start) as an mp4, and N_FRAMES keyframes spread over it."""
    clip = VideoRef(ref.path, ref.from_s + t0, ref.from_s + t1)
    frames, _ = read_clip(clip, sample_fps=VIDEO_FPS, max_side=MAX_SIDE)
    times = [round(t0 + i * (t1 - t0) / (N_FRAMES - 1), 2) for i in range(N_FRAMES)]
    keys = frames_at(ref, [min(t, t1 - 0.05) for t in times], max_side=MAX_SIDE)
    return mp4_bytes(frames), [jpeg_bytes(k) for k in keys], [round(t - t0, 1) for t in times]


class Recorder:
    """A Token Factory chat that also keeps each call's usage and caches answers by request content."""

    def __init__(self, key: str | None):
        self.key = key
        self.usage: list[dict] = []
        (CACHE / "super_vl").mkdir(parents=True, exist_ok=True)

    def __call__(self, messages: list[dict], model: str) -> str:
        body = token_factory_body(messages, model, temperature=0.0, max_tokens=60)
        digest = hashlib.sha256(json.dumps([PROMPT_VERSION, body], sort_keys=True).encode("utf-8")).hexdigest()[:24]
        path = CACHE / "super_vl" / f"{digest}.json"
        if path.exists():
            hit = json.loads(path.read_text(encoding="utf-8"))
            self.usage.append({**hit["usage"], "cached": True})
            return hit["text"]
        if self.key is None:
            raise RuntimeError("NEBIUS_API_KEY is not set and this request is not cached")
        req = urllib.request.Request(BASE_URL + "chat/completions", data=json.dumps(body).encode("utf-8"),
                                     headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        t = time.time()
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.loads(r.read().decode("utf-8"))
        text = token_factory_text(data)
        usage = {**(data.get("usage") or {}), "seconds": round(time.time() - t, 1)}
        path.write_text(json.dumps({"model": model, "prompt": PROMPT_VERSION, "text": text, "usage": usage}), encoding="utf-8")
        self.usage.append(usage)
        return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="build every request and print sizes; no network")
    args = ap.parse_args()
    key = None if args.dry_run else os.environ.get("NEBIUS_API_KEY")
    chat = Recorder(key)
    rows = []
    for ds in sorted({k.split("/")[0] for k in LABELS}):
        res = load(DATA / ds)
        cam = CAMERA.get(ds)
        curves = {j: load_curves(res.episodes, CACHE, j, cam) for j in ("robometer", "topreward")}
        verdicts = diagnose_batch(res.episodes, curves)
        flagged = {v.episode_id for v in verdicts if v.label != "合格"}
        want = {k for k in LABELS if k.startswith(ds + "/")}
        assert flagged == want, f"{ds}: flagged episodes differ from the labels: {sorted(flagged ^ want)}"
        by_id = {e.id: e for e in res.episodes}
        reqs = {r.episode_id: r for r in review_requests(verdicts, res.episodes)}
        for ep_id in sorted(want):
            lab = LABELS[ep_id]
            row = {"episode_id": ep_id, **{k: lab[k] for k in ("verdict", "human", "note")}}
            req = reqs.get(ep_id)
            ep = by_id[ep_id]
            refs = [(c, r) for c, r in ep.cameras.items() if Path(r.path).exists() and (cam is None or c == cam)]
            if req is None:
                rows.append({**row, "skipped": "no question for this finding"})
                continue
            row.update({"check": req.check, "t0": round(req.t0, 2), "t1": round(req.t1, 2), "question": req.question,
                        "choices": list(req.choices)})
            if not refs:
                rows.append({**row, "skipped": "no local video"})
                continue
            video, frames, times = media(refs[0][1], req.t0, req.t1)
            row["camera"] = refs[0][0]
            row["request_kb"] = {"video": round(len(video) * 4 / 3 / 1024, 1), "frames": round(sum(map(len, frames)) * 4 / 3 / 1024, 1)}
            if not args.dry_run:
                for mode, kw in (("video", {"video": video}), ("frames", {"frames": frames, "frame_times": times})):
                    try:
                        r = review(req, chat, **kw)
                        row[mode] = {"answer": r.answer, "outcome": r.outcome, "problems": r.problems, "usage": chat.usage[-len(r.attempts):]}
                    except (urllib.error.URLError, RuntimeError, KeyError) as exc:
                        row[mode] = {"answer": None, "outcome": "error", "error": f"{type(exc).__name__}: {exc}"[:300]}
            rows.append(row)
            print(f"{ep_id.split('__')[-1]:45s} {lab['verdict']:6s} {lab['human']:14s} {req.check:15s} "
                  f"video {row.get('video', {}).get('answer')!s:22s} frames {row.get('frames', {}).get('answer')!s:22s} "
                  f"{row['request_kb']['video']:.0f}+{row['request_kb']['frames']:.0f} KB", flush=True)
    if args.dry_run:
        asked = [r for r in rows if "request_kb" in r]
        print(f"{len(rows)} flagged, {len(asked)} with local video and a question, "
              f"{sum(r['request_kb']['video'] + r['request_kb']['frames'] for r in asked) / 1024:.1f} MB of requests")
        return 0
    groups = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        if "video" not in r:
            continue
        g = f"{r['verdict']}/{r['human']}" + ("/video_length" if r["check"] == "video_length" else "")
        for mode in ("video", "frames"):
            groups[(g, mode)][r[mode]["answer"] if r["check"] == "video_length" else r[mode]["outcome"]] += 1
    summary = {f"{g} [{m}]": dict(c) for (g, m), c in sorted(groups.items())}
    for k, v in summary.items():
        print(f"{k:45s} {v}")
    fresh = [u for u in chat.usage if not u.get("cached")]
    out = {"date": time.strftime("%Y-%m-%d %H:%M"), "model": MODEL, "prompt_version": PROMPT_VERSION,
           "base_url": BASE_URL, "video_fps": VIDEO_FPS, "max_side": MAX_SIDE, "keyframes": N_FRAMES,
           "calls": len(chat.usage), "new_calls": len(fresh),
           "prompt_tokens_new": sum(u.get("prompt_tokens", 0) for u in fresh),
           "completion_tokens_new": sum(u.get("completion_tokens", 0) for u in fresh),
           "summary": summary, "episodes": rows}
    (ROOT / "results" / "super_vl_review.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved results/super_vl_review.json;", len(fresh), "new calls,", out["prompt_tokens_new"], "prompt tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
