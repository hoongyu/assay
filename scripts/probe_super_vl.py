"""Can Nemotron 3.5 Super VL on Nebius Token Factory read our robot video? Four small calls:

1. text only: the model ID is served and the key works
2. one keyframe as image_url: prompt tokens for one image
3. eight keyframes as image_url: how tokens grow with frames
4. the whole clip as video_url (base64 mp4): does Token Factory accept video at all

Each call asks the same fixed question about the ep2 regrasp (towel folding, right gripper loses the towel
corner at 12.8 s and grabs it again at 13.5 s; in the 11 to 15.5 s clip that is 1.8 s and 2.5 s) and wants a
fixed JSON answer. Prints HTTP status, usage and the start of each answer, and writes
results/probe_super_vl.json. The key is read from NEBIUS_API_KEY and never printed or saved.
The four calls together cost well under one US cent.

    uv run python scripts/probe_super_vl.py            # needs NEBIUS_API_KEY in this terminal
    uv run python scripts/probe_super_vl.py --dry-run  # builds the requests only, no network
"""

from __future__ import annotations

import argparse
import base64
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

from assay.keyframes import jpeg_data_uri

ROOT = Path(__file__).resolve().parents[1]
CLIP = ROOT / "reports" / "clips" / "fold_towel_ep2_regrasp_11s-15.5s.mp4"
FPS = 30
BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
MODEL = "nvidia/Nemotron-3.5-Super-VL"
QUESTION = (
    "This is a robot teleoperation recording of two arms folding a towel. Look at the right gripper (on the right "
    "side of the image) between 1.5 s and 3.0 s of the clip. Does it lose its grip on the towel and grasp the towel "
    "again? Answer with JSON only, no other text: "
    '{"regrasp": "yes" | "no" | "unclear", "gripper_visible": "yes" | "no"}'
)


def clip_frames(max_side: int = 448) -> list[np.ndarray]:
    """Every frame of the clip in decoding order. The clip's timestamps are broken (all 136 frames say 0.0 s),
    so frames are taken by position, not by time: 136 frames over 4.5 s, about 30 per second."""
    with av.open(str(CLIP)) as c:
        s = c.streams.video[0]
        w, h = s.codec_context.width, s.codec_context.height
        scale = min(1.0, max_side / max(w, h))
        tw, th = round(w * scale / 2) * 2, round(h * scale / 2) * 2
        return [f.reformat(width=tw, height=th).to_ndarray(format="rgb24") for f in c.decode(s)]


def keyframe_uris(frames: list[np.ndarray], n: int) -> list[str]:
    """n JPEG data URIs evenly spread over the clip, first and last frame included."""
    picks = [round(i * (len(frames) - 1) / max(n - 1, 1)) for i in range(min(n, len(frames)))]
    return [jpeg_data_uri(frames[i]) for i in picks]


def video_data_uri(frames: list[np.ndarray]) -> str:
    """The clip re-encoded as H.264 at 30 fps with clean timestamps, as a base64 mp4 data URI."""
    buf = io.BytesIO()
    with av.open(buf, "w", format="mp4") as out:
        s = out.add_stream("libx264", rate=FPS)
        s.height, s.width = frames[0].shape[:2]
        s.pix_fmt = "yuv420p"
        for img in frames:
            for p in s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")):
                out.mux(p)
        for p in s.encode(None):
            out.mux(p)
    return "data:video/mp4;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def body(parts: list[dict]) -> dict:
    return {"model": MODEL, "temperature": 0.0, "max_tokens": 200,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": parts + [{"type": "text", "text": QUESTION}]}]}


def cases() -> list[tuple[str, dict]]:
    frames = clip_frames()
    one = [jpeg_data_uri(frames[round(1.8 * FPS)])]   # 12.8 s in the episode: the corner slips
    eight = keyframe_uris(frames, 8)
    return [
        ("text_only", {"model": MODEL, "temperature": 0.0, "max_tokens": 20,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "user", "content": "Reply with the single word: ready"}]}),
        ("one_image", body([{"type": "image_url", "image_url": {"url": u}} for u in one])),
        ("eight_images", body([{"type": "image_url", "image_url": {"url": u}} for u in eight])),
        ("video_mp4", body([{"type": "video_url", "video_url": {"url": video_data_uri(frames)}}])),
    ]


def post(payload: dict, key: str) -> dict:
    req = urllib.request.Request(BASE_URL + "chat/completions", data=json.dumps(payload).encode("utf-8"),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.loads(r.read().decode("utf-8"))
        choice = data["choices"][0]
        return {"status": 200, "seconds": round(time.time() - t0, 1), "usage": data.get("usage"),
                "finish_reason": choice.get("finish_reason"), "answer": (choice["message"].get("content") or "")[:600]}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "seconds": round(time.time() - t0, 1),
                "error": e.read().decode("utf-8", "replace")[:600]}
    except Exception as e:  # network, timeout
        return {"status": None, "seconds": round(time.time() - t0, 1), "error": f"{type(e).__name__}: {e}"[:600]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="build the four requests and print their sizes only")
    args = ap.parse_args()
    built = cases()
    if args.dry_run:
        for name, payload in built:
            print(f"{name:13s} request {len(json.dumps(payload)) / 1024:7.1f} KB")
        return 0
    key = os.environ.get("NEBIUS_API_KEY")
    if not key:
        print("NEBIUS_API_KEY is not set in this terminal")
        return 2
    out = {"date": time.strftime("%Y-%m-%d %H:%M"), "base_url": BASE_URL, "model": MODEL, "clip": CLIP.name,
           "question": QUESTION, "calls": {}}
    for name, payload in built:
        res = post(payload, key)
        res["request_kb"] = round(len(json.dumps(payload)) / 1024, 1)
        out["calls"][name] = res
        tokens = (res.get("usage") or {}).get("prompt_tokens")
        print(f"{name:13s} HTTP {res['status']}  {res['seconds']:5.1f} s  prompt_tokens {tokens}  "
              f"{(res.get('answer') or res.get('error') or '').strip()[:160]!r}")
    path = ROOT / "results" / "probe_super_vl.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
