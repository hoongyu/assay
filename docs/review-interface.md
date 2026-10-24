# Video review interface (Nemotron 3.5 Super VL)

For the browser demo and any other caller. Module: `assay/review.py` (standard library only: no `av`, no network). Request body helpers: `assay/summary.py`. Prompt version `"1"`, model `nvidia/Nemotron-3.5-Super-VL` (constants `PROMPT_VERSION`, `MODEL`).

## Flow

1. `review_requests(verdicts, episodes) -> list[ReviewRequest]`: one request per DOUBT or REJECT episode that has a reviewable finding (order: regrasp, end_pose, empty_recording, progress_drop, wrong_task, video_length, too_short). PASS episodes get none.
2. The caller cuts the clip `[req.t0, req.t1]` (seconds from the episode start) out of the episode's video and sends it either as an mp4 (`video=bytes`) or as JPEG keyframes (`frames=[bytes, ...]`, `frame_times=[seconds from the clip start, ...]`). The command line uses 10 fps, long side 448, and 8 keyframes.
3. `review(req, chat, model=MODEL, *, video=None, frames=None, frame_times=None) -> ReviewResult`. `chat(messages, model) -> str` is injected. The answer must be JSON `{"answer": <one of req.choices>}`; anything else is asked once more, then recorded as `unclear`.
4. `apply_reviews(verdicts, results) -> list[Verdict]` attaches results to `Verdict.reviews`. Labels never change.

## Sending a request from the browser

```
body = token_factory_body(messages, MODEL, temperature=0.0, max_tokens=60)
POST https://api.tokenfactory.nebius.com/v1/chat/completions   (Authorization: Bearer <key>)
text = token_factory_text(response_json)   # raises Truncated when finish_reason == "length"
```

`messages` comes from `review_messages(req, frames, frame_times, video)` (called inside `review`): one user message whose content is a `video_url` part (`data:video/mp4;base64,...`) or `image_url` parts (`data:image/jpeg;base64,...`), then a text part with the question and the answer format.

## ReviewRequest and ReviewResult

| Field | Meaning |
|---|---|
| `episode_id`, `check` | which episode and which finding is reviewed |
| `t0`, `t1` | clip window in seconds from the episode start |
| `question` | English question generated from the rule (times are relative to the clip start) |
| `choices`, `confirms`, `clears` | allowed answers; answers that support or contradict the finding (`video_length` is descriptive: no `clears`) |
| `answer`, `outcome` | the accepted answer or `None`; `confirms`, `clears` or `unclear` |
| `problems`, `attempts` | why an answer was refused; raw answers |

Measured on 2026-10-20 (`results/probe_super_vl.json`): Token Factory accepts `video_url`; a 448×336 keyframe costs about 300 prompt tokens, a 4.5 s clip about 880 in total.
