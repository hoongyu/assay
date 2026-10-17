---
title: Assay 甄数
emoji: 🔍
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Auto QC for robot teleoperation demonstrations
---

# Assay 甄数 · online demo

Pick a public LeRobot v3 dataset on the Hugging Face Hub, or upload your own (LeRobot v3 as a zip, or a robomimic / RoboMIND HDF5 file). Assay marks every demonstration pass, review or reject, with the reason, the readings it relied on and the time span, and returns its offline HTML report.

This demo runs the joint-signal rules on CPU only; the zero-shot video judges need a GPU and are not part of it. Uploads are processed in a temporary folder and deleted after the report is built.

Code: https://github.com/hoongyu/assay (Apache-2.0)
