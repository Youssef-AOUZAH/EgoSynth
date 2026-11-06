#!/usr/bin/env python3
"""EgoSynth: synthetic egocentric video dataset generation for VLA pretraining.

Three phases, run sequentially so that only one large model is on the GPU at a time:

  1. Gemma 3 12B          expands each short task phrase into a detailed video prompt
  2. LTX-2.3 22B          renders N candidate clips per task (default 3 seeds)
  3. Qwen2.5-VL-72B (AWQ) scores every candidate 0-100; the best is kept, the rest deleted

Usage:
    python generate_dataset.py --csv_file examples/tasks.csv --output_dir my_dataset --run_local
    python generate_dataset.py --csv_file tasks.csv --output_dir my_dataset      # submits a Slurm job
    python generate_dataset.py --csv_file tasks.csv --dry_run                    # no GPU needed

See README.md for installation and the full list of options.
"""

import argparse
import csv
import gc
import importlib.util
import json
import os
import shlex
import subprocess
import sys

from referee import REFEREE_SYSTEM_PROMPT, REFEREE_USER_PROMPT, parse_score

# LTX-2.3 weight file names, relative to --ltx_dir (see README for the download command).
LTX_CHECKPOINT = "ltx-2.3-22b-dev.safetensors"
LTX_DISTILLED_LORA = "ltx-2.3-22b-distilled-lora-384.safetensors"
LTX_DISTILLED_LORA_STRENGTH = "0.8"
LTX_SPATIAL_UPSAMPLER = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
LTX_MODULE = "ltx_pipelines.ti2vid_two_stages"

MAX_SEEDS = 9  # seeds are derived as task_index * 10 + seed_index, which needs seed_index < 10
MIN_VALID_VIDEO_BYTES = 10_000

EXPANSION_SYSTEM_PROMPT = (
    "You are a professional cinematographer and robotics expert. "
    "I will give you a short task description for a robotic arm. "
    "You must instantly expand it into a highly detailed, extremely cinematic, photorealistic prompt for a text-to-video model. "
    "Focus on the exact hand movements, lighting, textures, camera angle (first-person), and physics. "
    "DO NOT output anything else except the final expanded prompt. Be extremely descriptive."
)


# --------------------------------------------------------------------------- CLI

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="EgoSynth: end-to-end synthetic egocentric video pipeline")
    p.add_argument("--csv_file", default="tasks.csv", help="CSV of task phrases (column 'task' or 'prompt', else first column)")
    p.add_argument("--output_dir", default="generated_dataset", help="Output directory")
    p.add_argument("-n", "--chunk_size", type=int, default=10,
                   help="Tasks processed per end-to-end chunk (models are reloaded per chunk, so larger = fewer reloads)")
    p.add_argument("--num_seeds", type=int, default=3, help="Candidate clips generated per task")
    p.add_argument("--num_frames", type=int, default=121, help="Frames per clip")
    p.add_argument("--keep_all", action="store_true", help="Keep all candidates instead of deleting the lower-scoring ones")
    p.add_argument("--run_local", action="store_true", help="Run directly instead of submitting a Slurm job")
    p.add_argument("--dry_run", action="store_true", help="Print what would run (tasks, commands, Slurm script) and exit")
    p.add_argument("--offline", action="store_true", help="Set HF_HUB_OFFLINE=1 (for compute nodes without internet)")

    m = p.add_argument_group("models")
    m.add_argument("--gemma_path", default=os.getenv("GEMMA_PATH", "models/gemma-3-12b"),
                   help="Gemma 3 12B instruct (local dir or HF id); also used as LTX text encoder [env GEMMA_PATH]")
    m.add_argument("--ltx_dir", default=os.getenv("LTX_CHECKPOINT_DIR", "checkpoints"),
                   help="Directory holding the LTX-2.3 weights [env LTX_CHECKPOINT_DIR]")
    m.add_argument("--qwen_path", default=os.getenv("QWEN_MODEL_PATH", "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"),
                   help="Qwen2.5-VL referee (the AWQ build fits on one 80 GB GPU) [env QWEN_MODEL_PATH]")

    s = p.add_argument_group("slurm (only used when submitting a job)")
    s.add_argument("--python", default="python", help="Python executable the Slurm job should use (e.g. a conda env's python)")
    s.add_argument("--slurm_account", default=os.getenv("SLURM_ACCOUNT"), help="Slurm account [env SLURM_ACCOUNT]")
    s.add_argument("--slurm_partition", default="gpu")
    s.add_argument("--slurm_qos", default=None)
    s.add_argument("--slurm_time", default="24:00:00")
    s.add_argument("--slurm_mem", default="128G")
    s.add_argument("--slurm_cpus", type=int, default=16)
    s.add_argument("--slurm_setup", default=None,
                   help="Shell commands run before python, e.g. 'module load Anaconda3; source activate egosynth'")

    # Notebook-friendly: ignore Jupyter's hidden arguments (e.g. -f kernel.json).
    try:
        get_ipython  # noqa: F821
        args, _ = p.parse_known_args([])
        print(f"[NOTE] Running in Jupyter/IPython with default arguments: csv_file='{args.csv_file}', output_dir='{args.output_dir}'")
    except NameError:
        args = p.parse_args(argv)

    if not 1 <= args.num_seeds <= MAX_SEEDS:
        p.error(f"--num_seeds must be between 1 and {MAX_SEEDS}")
    return args


# ----------------------------------------------------------------------- helpers

def read_tasks(csv_path):
    """Return {task_id: phrase}. The first row is a header; column 'task' or 'prompt' is used if present."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return {}
    header = [h.strip().lower() for h in rows[0]]
    col = next((header.index(name) for name in ("task", "prompt") if name in header), 0)
    tasks = {}
    for idx, row in enumerate(rows[1:]):
        if len(row) > col and row[col].strip():
            tasks[f"task_{idx:03d}"] = row[col].strip()
    return tasks


def task_index(task_id):
    return int(task_id.split("_")[1])


def candidate_name(seed_idx):
    return f"video_seed_{seed_idx}.mp4"


def ltx_command(args, prompt, output_path, seed):
    d = args.ltx_dir
    return [
        sys.executable, "-m", LTX_MODULE,
        "--checkpoint-path", os.path.join(d, LTX_CHECKPOINT),
        "--distilled-lora", os.path.join(d, LTX_DISTILLED_LORA), LTX_DISTILLED_LORA_STRENGTH,
        "--spatial-upsampler-path", os.path.join(d, LTX_SPATIAL_UPSAMPLER),
        "--gemma-root", args.gemma_path,
        "--prompt", prompt,
        "--output-path", output_path,
        "--num-frames", str(args.num_frames),
        "--seed", str(seed),
    ]


def build_slurm_script(args):
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    lines = ["#!/bin/bash", "#SBATCH --job-name=egosynth"]
    if args.slurm_account:
        lines.append(f"#SBATCH --account={args.slurm_account}")
    lines += [f"#SBATCH --partition={args.slurm_partition}"]
    if args.slurm_qos:
        lines.append(f"#SBATCH --qos={args.slurm_qos}")
    lines += [
        f"#SBATCH --mem={args.slurm_mem}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={args.slurm_cpus}",
        "#SBATCH --gres=gpu:1",
        f"#SBATCH --time={args.slurm_time}",
        "#SBATCH --output=egosynth_%j.out",
        "#SBATCH --error=egosynth_%j.err",
        "",
    ]
    if args.slurm_setup:
        lines += [args.slurm_setup, ""]
    forwarded = [
        "--csv_file", os.path.abspath(args.csv_file),
        "--output_dir", os.path.abspath(args.output_dir),
        "-n", str(args.chunk_size),
        "--num_seeds", str(args.num_seeds),
        "--num_frames", str(args.num_frames),
        "--gemma_path", args.gemma_path,
        "--ltx_dir", args.ltx_dir,
        "--qwen_path", args.qwen_path,
    ]
    if args.keep_all:
        forwarded.append("--keep_all")
    if args.offline:
        forwarded.append("--offline")
    lines.append(f"cd {shlex.quote(repo_dir)}")
    lines.append(" ".join([shlex.quote(args.python), "generate_dataset.py", *map(shlex.quote, forwarded), "--run_local"]))
    return "\n".join(lines) + "\n"


def check_environment(args):
    """Fail early (before hours of GPU time) if something required is missing."""
    problems = []
    for mod in ("torch", "transformers", "qwen_vl_utils", "ltx_pipelines"):
        if importlib.util.find_spec(mod) is None:
            problems.append(f"Python package '{mod}' is not installed (see README > Installation).")
    for name in (LTX_CHECKPOINT, LTX_DISTILLED_LORA, LTX_SPATIAL_UPSAMPLER):
        if not os.path.isfile(os.path.join(args.ltx_dir, name)):
            problems.append(f"Missing LTX weight: {os.path.join(args.ltx_dir, name)}")
    if problems:
        print("[ERROR] Environment check failed:\n  - " + "\n  - ".join(problems))
        sys.exit(1)


def free_gpu(*objs):
    import torch
    del objs
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------- phase 1

def expand_prompts(chunk, args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    todo = {tid: raw for tid, raw in chunk.items()
            if not os.path.exists(os.path.join(args.output_dir, tid, "prompt_info.json"))}
    if not todo:
        return

    print("\n--- [PHASE 1] Gemma 3 prompt expansion ---")
    tokenizer = AutoTokenizer.from_pretrained(args.gemma_path)
    model = AutoModelForCausalLM.from_pretrained(args.gemma_path, device_map="auto", torch_dtype=torch.bfloat16)

    for tid, raw in todo.items():
        task_dir = os.path.join(args.output_dir, tid)
        os.makedirs(task_dir, exist_ok=True)
        print(f"Expanding: {raw[:60]}")
        msgs = [{"role": "user", "content": f"{EXPANSION_SYSTEM_PROMPT}\n\nTask: '{raw}'"}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        out = model.generate(**inputs, max_new_tokens=220, temperature=0.3, do_sample=True,
                             pad_token_id=tokenizer.eos_token_id)
        enhanced = tokenizer.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
        with open(os.path.join(task_dir, "prompt_info.json"), "w", encoding="utf-8") as f:
            json.dump({"video_id": tid, "raw_csv_prompt": raw, "enhanced_prompt": enhanced}, f, indent=4)

    free_gpu(model, tokenizer)


# ---------------------------------------------------------------------- phase 2

def synthesize_videos(chunk, args):
    print("\n--- [PHASE 2] LTX-2.3 video synthesis ---")
    for tid in chunk:
        task_dir = os.path.join(args.output_dir, tid)
        prompt_file = os.path.join(task_dir, "prompt_info.json")
        if not os.path.exists(prompt_file) or os.path.exists(os.path.join(task_dir, "eval_results.json")):
            continue
        with open(prompt_file, encoding="utf-8") as f:
            enhanced = json.load(f)["enhanced_prompt"]

        for seed_idx in range(1, args.num_seeds + 1):
            out_path = os.path.join(task_dir, candidate_name(seed_idx))
            if os.path.exists(out_path) and os.path.getsize(out_path) > MIN_VALID_VIDEO_BYTES:
                continue
            print(f"Generating {tid} -> seed {seed_idx}/{args.num_seeds}")
            cmd = ltx_command(args, enhanced, out_path, task_index(tid) * 10 + seed_idx)
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] LTX generation failed for {tid} seed {seed_idx}: {e}")


# ---------------------------------------------------------------------- phase 3

def referee_video(model, processor, process_vision_info, video_path):
    import torch

    msgs = [
        {"role": "system", "content": REFEREE_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "video", "video": video_path, "max_pixels": 360 * 420, "fps": 2.0},
            {"type": "text", "text": REFEREE_USER_PROMPT},
        ]},
    ]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(msgs)
    inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=256)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def run_referee(chunk, args):
    todo = [tid for tid in chunk if not os.path.exists(os.path.join(args.output_dir, tid, "eval_results.json"))]
    if not todo:
        return

    print("\n--- [PHASE 3] Qwen2.5-VL referee & cleanup ---")
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.qwen_path, torch_dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(args.qwen_path)

    for tid in todo:
        task_dir = os.path.join(args.output_dir, tid)
        scores, critiques, unparsed = {}, {}, []
        for seed_idx in range(1, args.num_seeds + 1):
            name = candidate_name(seed_idx)
            path = os.path.join(task_dir, name)
            if not os.path.exists(path):
                continue
            reply = referee_video(model, processor, process_vision_info, path)
            score = parse_score(reply)
            critiques[name] = reply
            if score is None:
                unparsed.append(name)
                print(f"[{tid}] {name}: could not read a score from the referee reply (excluded from selection)")
            else:
                scores[name] = score
                print(f"[{tid}] {name}: score {score}")

        if not critiques:
            continue  # nothing was generated for this task; leave it for a re-run

        winner = max(scores, key=scores.get) if scores else None  # ties go to the lowest seed
        with open(os.path.join(task_dir, "eval_results.json"), "w", encoding="utf-8") as f:
            json.dump({"winner": winner, "score": scores.get(winner), "all_scores": scores,
                       "unparsed": unparsed, "critiques": critiques}, f, indent=4)

        if winner and not args.keep_all:
            for name in critiques:
                if name != winner and os.path.exists(os.path.join(task_dir, name)):
                    os.remove(os.path.join(task_dir, name))
                    print(f"   -> discarded {name}")

    free_gpu(model, processor)


# ------------------------------------------------------------------------- main

def main(argv=None):
    args = parse_args(argv)

    if not os.path.exists(args.csv_file):
        print(f"[ERROR] CSV file not found: {args.csv_file}")
        sys.exit(1)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    if not args.run_local:
        script = build_slurm_script(args)
        if args.dry_run:
            print("[DRY RUN] Slurm script that would be submitted:\n")
            print(script)
            return
        with open("auto_submit.slurm", "w", encoding="utf-8") as f:
            f.write(script)
        print("[INFO] Wrote auto_submit.slurm, submitting with sbatch...")
        try:
            subprocess.run(["sbatch", "auto_submit.slurm"], check=True)
            print("[SUCCESS] Job submitted. Monitor it with 'squeue --me'.")
            return
        except FileNotFoundError:
            print("[WARNING] 'sbatch' not found (not on an HPC login node?). Continuing locally.")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Slurm submission failed: {e}")
            return

    tasks = read_tasks(args.csv_file)
    if not tasks:
        print(f"[ERROR] No tasks found in {args.csv_file} (expected a header row plus at least one task).")
        sys.exit(1)

    if args.dry_run:
        first_id, first_prompt = next(iter(tasks.items()))
        print(f"[DRY RUN] {len(tasks)} tasks, chunks of {args.chunk_size}, {args.num_seeds} candidates each.")
        for tid, raw in list(tasks.items())[:5]:
            print(f"  {tid}: {raw}")
        print("\nExample LTX command (first task, seed 1):")
        print("  " + " ".join(shlex.quote(c) for c in
                              ltx_command(args, "<expanded prompt>", os.path.join(args.output_dir, first_id, candidate_name(1)),
                                          task_index(first_id) * 10 + 1)))
        return

    check_environment(args)
    os.makedirs(args.output_dir, exist_ok=True)

    items = list(tasks.items())
    print(f"[INFO] {len(items)} tasks, chunks of {args.chunk_size}.")
    for start in range(0, len(items), args.chunk_size):
        chunk = dict(items[start:start + args.chunk_size])
        print(f"\n=== CHUNK: tasks {start} to {start + len(chunk) - 1} ===")
        expand_prompts(chunk, args)
        synthesize_videos(chunk, args)
        run_referee(chunk, args)

    print("\n[SUCCESS] Pipeline complete. The curated dataset is in", args.output_dir)


if __name__ == "__main__":
    main()
