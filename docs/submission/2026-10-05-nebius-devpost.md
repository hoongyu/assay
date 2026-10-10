# Devpost submission draft: Nebius × NVIDIA Global AI Hackathon, Physical AI track

Draft for the user to paste into Devpost; the user submits. Deadline: 2026-10-30, 10:00 Pacific. Before submitting: make the GitHub repository public, upload the demo video to YouTube (under 3 minutes, at least 1 minute showing the software running), and fill in the team names.

---

**Project name:** Assay

**Tagline:** Automatic QA for robot teleoperation data: every episode gets PASS, DOUBT or REJECT, with the reason and the evidence.

## Inspiration

Robot foundation models are trained on teleoperation demonstrations, and in China this data is now produced in dedicated collection sites: at least 15 large ones, typically with a hundred or more robots each. Beijing Humanoid's site sells its data at 500 to 1,000 CNY per hour. QA and labeling still rely largely on manual work, and the head of one large site estimated that of 100 hours recorded, maybe 50 come out usable. We wanted a QA tool that a collection site can run on its own machines, because its data is its product, and that tells operators what they did wrong, not just which episodes were thrown away.

The problem is not only Chinese. When we ran Assay on five public LeRobot datasets on Hugging Face, published for training with no episode marked as failed, it rejected 6 of 308 episodes, and all 6 were unusable: recordings under one second, an arm that never moves, a person resetting the scene. Of the 33 more it sent to a person, 12 were real problems. Anyone training on community datasets inherits these episodes.

## What it does

`assay check <dataset>` reads a batch of episodes (LeRobot v3, RoboMIND HDF5 or RoboVAD) and writes a single offline HTML report:

- Every episode is judged PASS, DOUBT or REJECT. Rejection is conservative: an objective hard rule (missing readings, dropped frames, too short), or two independent sources pointing at the same problem.
- Each flagged episode gets a card: the problem class, the time span, the evidence, the progress curve and keyframes. For a regrasp, the keyframes are taken at the gripper's switch times.
- The report comes in English or Chinese (`--lang en`). Every piece of evidence is written in both languages by the code, from the same numbers, without a model. Example from a public towel-folding dataset: the right gripper switched between open and closed 6 times where the same task usually switches at most 4 times, and two of the switch times, 12.8 s and 13.5 s, are where the video shows it losing the towel corner and grasping it again.
- With `--summary`, NVIDIA Nemotron 3.5 Lightning on Nebius Token Factory writes a short batch summary for the QA lead and the operators, in English and Chinese. Assay fills in every number, and the tips for the operators come only from a fixed list we wrote, one per problem class.

Try it in two minutes without a GPU: the repository README has four commands that download the public datasets' joint data (about 73 MB) and produce a report. The joint checks alone flag 5 of 56 cube-stacking episodes, and all five are real problems.

## How we built it

- Joint rules on CPU: integrity checks, long pauses, empty recordings, end poses that do not look finished, extra motion, regrasps (gripper open and close cycles compared with the same task's batch) and unusually long episodes.
- Two zero-shot video judges, Robometer-4B and TOPReward on Qwen3-VL-4B, run on Colab GPUs with cached results.
- Batch summary: Nemotron sees only aggregate statistics, never joint data or video. It writes prose with placeholders like `{n_doubt}` and `{advice_1}`; Assay rejects any digit the model writes on its own, retries once, then falls back to a template. `Assay summary-stats` counts how often each of the three happened. The request is about 15 lines of Python standard library against Token Factory's OpenAI-compatible endpoint, and answers are cached by their inputs.
- Three exams on real data, with thresholds chosen on dev sets and every number reported with its denominator.

Open-source tools such as Loop-Engineering-for-VLA and hflow already audit LeRobot datasets, with similarly conservative drop or review verdicts. What Assay adds: the failure cause with its evidence and timing, written so it can go back to the operator; detection measured on a real QA team's labels; and a rejection only when a hard rule fires or two independent sources agree.

## Results

- Public demonstration datasets (5 LeRobot datasets, 308 episodes): 6 rejected, all 6 real problems (5 checked on video). 33 sent to a person: 12 real problems, 12 false positives, 9 not checked yet.
- Injected defects on held-out datasets and tasks: cut endings 197/201 detected and frozen frames 199/201, with 0/202 clean episodes flagged. The cut-ending row is our second look at the test set: the first look flagged 11/202 clean towel-folding episodes, so we added bimanual dev data, changed the scaling and chose the threshold on dev again.
- RoboVAD real anomalies (no model trained on RoboVAD; thresholds chosen on its train split): 247/387 anomalous episodes flagged against 33/242 normal ones.
- RoboMIND, 1,678 episodes rejected by Beijing Humanoid's QA, 1,310 of them on real robots: of the 162 regrasp rejections Assay can compare, the regrasp rule, tuned on another dataset and not retuned, flags 99 (61%; real robots only, 99/159). It also flags 7% of same-task episodes rejected for other reasons, 21% on the dual-arm tasks.
- Batch summaries from Nemotron 3.5 Lightning on Token Factory (four public batches, English and Chinese): 8/8 passed the number check on the first try, with no retries and no template fallbacks. Reading them, 1 of 8 still used a placeholder twice in one muddled sentence and paraphrased a tip, which the number check cannot catch.

## Challenges we ran into

- Smoothness metrics did not work. On RoboMIND's dev split, episodes labeled "unsmooth" had fewer stops and movement segments than other rejections and no more jerk (AUC 0.31 to 0.45); on test the same metrics were near chance (0.45 to 0.56). Only the long tail of episode duration separated the classes, so we kept a duration rule, which catches 14% on test, and say so.
- The first two Nemotron runs failed our number check. With reasoning on (the default), the reasoning trace came back inside the answer text and ran into the token limit before the note was written. Logging every raw attempt showed the cause; `enable_thinking: false` fixed it, and both languages then passed on the first try.
- RoboMIND's files are 440.9 GB, mostly images. We read only the joint datasets out of each HDF5 file with HTTP range requests: 1.75 GB read in 29 minutes.

## Accomplishments that we're proud of

- A regrasp rule chosen on one benchmark that still finds 99 of 162 comparable regrasps in another lab's QA labels without retuning. It is weakest on dual-arm tasks, where it also flags 21% of other rejections.
- An LLM summary that never writes its own digits: the model writes placeholders and Assay fills in every number.
- Honest evaluation: dev and test splits fixed before looking, second looks at a test set disclosed, failures reported.

## What we learned

- On real QA labels, "obvious" motion-quality metrics can score below chance on one split and at chance on the next; measure before shipping.
- With reasoning models behind an OpenAI-compatible API, check where the reasoning goes before parsing the answer.

## What's next

- A pilot with one collection site, to measure how much review time Assay saves and whether per-operator feedback lowers the rejection rate. The business model follows how sites sell data: they charge per hour of data, and Assay would charge per hour checked (7 CNY in our model, under 1.5% of what Beijing Humanoid charges for an hour of data).
- A video model that can answer specific questions for the problems joints cannot see: unwanted contact, collision before grasp, objects knocked away.

## Built with

Python, NumPy, h5py, PyArrow, PyAV, Nebius Token Factory, NVIDIA Nemotron 3.5 Lightning, Robometer-4B, Qwen3-VL-4B, LeRobot, Hugging Face Hub, Google Colab.

## Feedback on Nebius and NVIDIA tools

**What we used:** Nebius Token Factory's OpenAI-compatible API with NVIDIA Nemotron 3.5 Lightning, to write the batch summary at runtime.

**What worked well:**
- The OpenAI-compatible endpoint: about 15 lines of standard-library Python, no SDK needed.
- Nemotron 3.5 Lightning has open weights (OpenMDW-1.1), which matters for our users: a collection site can later run the same model on its own machines, so the summary works without data leaving the site.
- With reasoning off, the model followed the placeholder format in both Chinese and English on the first attempt.

**What could be better:**
- The Token Factory quickstart shows one DeepSeek example; it does not list the NVIDIA model ids or explain how to turn reasoning on or off.
- With reasoning on, Nemotron's reasoning trace comes back inside `content`, with no `<think>` tags and no separate field, so it is easy to mistake for the answer. We found the `chat_template_kwargs: {"enable_thinking": false}` switch on the Hugging Face model card; Token Factory honors it, but the Token Factory docs do not mention it.
- Since `nvidia/Cosmos3-Super-Reasoner` left Token Factory serverless on 2026-08-31, there is no NVIDIA video model on serverless. For a Physical AI track that is a gap: our first plan was Cosmos 3 explaining video clips, but that now needs a paid GPU endpoint, and the online credits only cover Token Factory.

**Onboarding:** The first call worked as soon as the API key was set; the time went into the reasoning-trace issue above.

**Would we use it again:** Yes for text generation. For video understanding we would come back if Cosmos returns to serverless or gets a low-cost dedicated option.
