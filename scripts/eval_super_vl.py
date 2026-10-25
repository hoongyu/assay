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
import collections
import json
import os
import sys
import time
import urllib.error
from pathlib import Path

from assay.diagnose import diagnose_batch
from assay.ingest import load
from assay.judges.cache import load_curves
from assay.review import MODEL, PROMPT_VERSION, review, review_requests
from assay.review_run import MAX_SIDE, N_FRAMES, VIDEO_FPS, CachedChat, clip_media
from assay.summary import BASE_URL

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = ROOT / "cache"
LABELS = json.loads((ROOT / "configs" / "public_review_labels.json").read_text(encoding="utf-8"))["labels"]
CAMERA = {"CoRL2026-CSI__SO101-DualArm-teleop_fold_towel_100epi": "observation.images.left_top"}  # the camera the judges used


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="build every request and print sizes; no network")
    args = ap.parse_args()
    key = None if args.dry_run else os.environ.get("NEBIUS_API_KEY")
    chat = CachedChat(key, CACHE)
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
            video, frames, times = clip_media(refs[0][1], req.t0, req.t1)
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
