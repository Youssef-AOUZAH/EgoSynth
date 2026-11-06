"""Original hard-coded three-stage batch script (kept for reference).

This is the script used for the 300-task batch of the original project: Gemma-3-12B
prompt expansion -> LTX-2.3 22B (3 seeds per task) -> Qwen2.5-VL-72B AWQ scoring.
It is more environment-specific than ../generate_dataset.py (fixed paths, a
kitchen-specific expansion prompt, a shorter referee prompt, and compatibility
patches for the HPC environment it ran in). Prefer generate_dataset.py for new runs.

Input: a JSON file {"task_000": "open drawer", "task_001": ...} (default
batch_300_prompts.json, override with the first command-line argument).
Run it from a directory containing checkpoints/ and gemma-3-12b-it/.
"""

import ast
import gc
import json
import os
import re
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Compatibility patches for the AST / AWQ / transformers versions on the original cluster.
for old_node in ["Num", "Str", "Bytes", "NameConstant", "Ellipsis"]:
    if not hasattr(ast, old_node):
        setattr(ast, old_node, ast.Constant)
if not hasattr(ast.Constant, "n"):
    ast.Constant.n = property(lambda self: self.value)
if not hasattr(ast.Constant, "s"):
    ast.Constant.s = property(lambda self: self.value)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import transformers  # noqa: E402
import transformers.activations  # noqa: E402

for attr in ["PytorchGELUTanh", "NewGELUActivation", "GELUActivation"]:
    if not hasattr(transformers.activations, attr):
        class CompatGELU(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.act = nn.GELU(approximate="tanh" if "Tanh" in attr else "none")

            def forward(self, x):
                return self.act(x)

        setattr(transformers.activations, attr, CompatGELU)

sys.modules["torchcodec"] = None
from transformers import (AutoProcessor, AutoTokenizer, AutoModelForCausalLM,  # noqa: E402
                          Qwen2_5_VLForConditionalGeneration)
from qwen_vl_utils import process_vision_info  # noqa: E402

MASTER_DIR = "pipeline_test_300"
PROMPTS_JSON = sys.argv[1] if len(sys.argv) > 1 else "batch_300_prompts.json"
os.makedirs(MASTER_DIR, exist_ok=True)

with open(PROMPTS_JSON, "r", encoding="utf-8") as f:
    tasks_dict = json.load(f)

print("=" * 70)
print(f" STARTING AUTOMATED VLA PIPELINE: {len(tasks_dict)} tasks (3 videos + CoT evaluation per task)")
print("=" * 70)

for vid_id in tasks_dict:
    os.makedirs(os.path.join(MASTER_DIR, vid_id), exist_ok=True)

# =====================================================================
# PHASE 1: GEMMA-3-12B PROMPT ENHANCEMENT
# =====================================================================
print("\n--- [PHASE 1] Gemma-3-12B prompt expansion ---")
need_gemma = any(not os.path.exists(os.path.join(MASTER_DIR, v, "prompt_info.json")) for v in tasks_dict)

if need_gemma:
    gemma_path = "gemma-3-12b-it"
    tokenizer = AutoTokenizer.from_pretrained(gemma_path)
    model = AutoModelForCausalLM.from_pretrained(gemma_path, device_map="auto", dtype=torch.bfloat16)

    sys_prompt = (
        "You are a biomechanics and cinematic lighting director. Expand this simple kitchen action description "
        "into a high-fidelity single paragraph for egocentric robotic VLA training video synthesis. Specify "
        "masculine hand skin texture, realistic grip pre-shaping, seamless contact physics without geometry clipping, "
        "and soft studio LED overhead fill lighting. Output ONLY the refined single paragraph without introductory filler."
    )

    for vid_id, raw_prompt in tasks_dict.items():
        task_dir = os.path.join(MASTER_DIR, vid_id)
        p_file = os.path.join(task_dir, "prompt_info.json")
        if os.path.exists(p_file):
            continue

        # Extract a clean mini-task phrase from the longer raw prompt.
        m = re.search(r"The hands ((?!enters|interact)[^.]+\.)", raw_prompt, re.IGNORECASE)
        mini_task_phrase = m.group(1).strip(".") if m else raw_prompt[:60].strip()
        print(f"[PHASE 1] Expanding '{mini_task_phrase}' for {vid_id}...")

        msgs = [{"role": "user", "content": f"{sys_prompt}\n\nTask: '{mini_task_phrase}'"}]
        input_ids = tokenizer.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True).to("cuda")
        outputs = model.generate(input_ids, max_new_tokens=220, temperature=0.3, do_sample=True,
                                 pad_token_id=tokenizer.eos_token_id)
        enhanced = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()

        with open(p_file, "w", encoding="utf-8") as pf:
            json.dump({"video_id": vid_id, "mini_task_phrase": mini_task_phrase,
                       "raw_csv_prompt": raw_prompt, "enhanced_prompt": enhanced}, pf, indent=4)

    print("[PHASE 1] Complete. Unloading Gemma to free VRAM for LTX...")
    del model, tokenizer, outputs, input_ids
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)
else:
    print("[PHASE 1] All prompt expansions already exist. Skipping Gemma.")

# =====================================================================
# PHASE 2: LTX-2.3 SYNTHESIS (3 VIDEOS PER TASK)
# =====================================================================
print("\n--- [PHASE 2] LTX-2.3 generation (3 seeds / task) ---")
for vid_id in sorted(tasks_dict.keys()):
    task_dir = os.path.join(MASTER_DIR, vid_id)
    with open(os.path.join(task_dir, "prompt_info.json"), "r", encoding="utf-8") as pf:
        enhanced_prompt = json.load(pf)["enhanced_prompt"]

    for seed_idx in range(1, 4):
        vid_path = os.path.join(task_dir, f"video_seed_{seed_idx}.mp4")
        if os.path.exists(vid_path) and os.path.getsize(vid_path) > 10000:
            continue

        print(f"[PHASE 2] Generating {vid_id} -> seed {seed_idx}/3...")
        cmd = [
            sys.executable, "-m", "ltx_pipelines.ti2vid_two_stages",
            "--checkpoint-path", "checkpoints/ltx-2.3-22b-dev.safetensors",
            "--distilled-lora", "checkpoints/ltx-2.3-22b-distilled-lora-384.safetensors", "0.8",
            "--spatial-upsampler-path", "checkpoints/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            "--gemma-root", "gemma-3-12b-it",
            "--prompt", enhanced_prompt,
            "--output-path", vid_path,
            "--num-frames", "121",
            "--seed", str(int(vid_id.split("_")[1]) * 10 + seed_idx),
        ]
        subprocess.run(cmd, check=True)

print("[PHASE 2] All video files synthesized.")

# =====================================================================
# PHASE 3: QWEN2.5-VL 72B CHAIN-OF-THOUGHT EVALUATION
# =====================================================================
print("\n--- [PHASE 3] Qwen2.5-VL-72B chain-of-thought realism evaluation ---")
need_qwen = any(not os.path.exists(os.path.join(MASTER_DIR, v, "eval_results.json")) for v in tasks_dict)

if need_qwen:
    qwen_path = os.getenv("QWEN_MODEL_PATH", "Qwen/Qwen2.5-VL-72B-Instruct-AWQ")
    if "HF_HOME_CUSTOM" in os.environ:
        os.environ["HF_HOME"] = os.environ["HF_HOME_CUSTOM"]

    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        qwen_path, dtype=torch.float16, device_map="auto", local_files_only=True)
    qwen_proc = AutoProcessor.from_pretrained(qwen_path, local_files_only=True)

    for vid_id in sorted(tasks_dict.keys()):
        task_dir = os.path.join(MASTER_DIR, vid_id)
        eval_file = os.path.join(task_dir, "eval_results.json")
        if os.path.exists(eval_file):
            continue

        with open(os.path.join(task_dir, "prompt_info.json"), "r", encoding="utf-8") as pf:
            p_data = json.load(pf)

        print(f"[PHASE 3] Evaluating candidates for {vid_id}...")
        results_summary = {
            "video_id": vid_id,
            "raw_prompt": p_data.get("raw_prompt") or p_data.get("raw_csv_prompt"),
            "enhanced_prompt": p_data["enhanced_prompt"],
            "evaluations": {},
            "winner": None,
        }

        best_score, best_vid = -1, None
        for seed_idx in range(1, 4):
            v_name = f"video_seed_{seed_idx}.mp4"
            v_path = os.path.join(task_dir, v_name)
            if not os.path.exists(v_path):
                continue

            msgs = [
                {"role": "system", "content": "You are a senior robotics simulation expert grading first-person hand manipulation training videos."},
                {"role": "user", "content": [
                    {"type": "video", "video": v_path, "max_pixels": 360 * 420, "fps": 2.0},
                    {"type": "text", "text": "Critically analyze this egocentric video for biomechanical realism. First, write 2 terse sentences observing hand fluidity, grasp contact accuracy, and shadow realism. Then, on a new line, assign a granular integer Realism Score between 0 and 100 based strictly on your observation. Format end of reply as: 'Score: [XX]'."},
                ]},
            ]
            text = qwen_proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            img_ins, vid_ins = process_vision_info(msgs)
            inputs = qwen_proc(text=[text], images=img_ins, videos=vid_ins, padding=True, return_tensors="pt").to("cuda")

            with torch.no_grad():
                out_ids = qwen_model.generate(**inputs, max_new_tokens=80)
            out_trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
            resp = qwen_proc.batch_decode(out_trimmed, skip_special_tokens=True)[0].strip()

            match = re.search(r"(?:Score|score)[:\s]*(\d+)", resp)
            numbers = re.findall(r"\d+", resp)
            score = int(match.group(1)) if match else (int(numbers[-1]) if numbers else 85)

            results_summary["evaluations"][v_name] = {"cot_critique": resp, "realism_score": score}
            print(f"   -> {v_name}: score {score} | {resp[:60]}...")

            if score > best_score:
                best_score, best_vid = score, v_name

        results_summary["winner"] = best_vid
        with open(eval_file, "w", encoding="utf-8") as ef:
            json.dump(results_summary, ef, indent=4)

    print("[PHASE 3] Qwen evaluation complete.")
