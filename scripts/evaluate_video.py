#!/usr/bin/env python3
"""Standalone referee: score every .mp4 in a directory with Qwen2.5-VL-72B (AWQ).

    python scripts/evaluate_video.py --video_dir path/to/videos
    python scripts/evaluate_video.py --video_dir path/to/videos --threshold 85 --sort

Writes evaluation_results.json next to the videos. With --sort, clips scoring at or
above the threshold are moved to good_videos/ and the rest to bad_videos/. Clips
whose score cannot be read are left in place and reported as unscored.
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from referee import REFEREE_SYSTEM_PROMPT, REFEREE_USER_PROMPT, parse_score  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"  # AWQ so that it fits on a single 80 GB GPU


def evaluate_video(model, processor, video_path):
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [
        {"role": "system", "content": REFEREE_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "video", "video": video_path, "max_pixels": 360 * 420, "fps": 2.0},
            {"type": "text", "text": REFEREE_USER_PROMPT},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=256)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def main():
    parser = argparse.ArgumentParser(description="Score generated videos with a Qwen2.5-VL referee")
    parser.add_argument("--video_dir", required=True, help="Directory containing .mp4 files")
    parser.add_argument("--model", default=os.getenv("QWEN_MODEL_PATH", DEFAULT_MODEL))
    parser.add_argument("--threshold", type=int, default=85, help="Minimum score to count as good (with --sort)")
    parser.add_argument("--sort", action="store_true", help="Move clips into good_videos/ and bad_videos/")
    args = parser.parse_args()

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print(f"Loading {args.model} ...")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
        processor = AutoProcessor.from_pretrained(args.model)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Could not load {args.model}: {e}\nCheck that transformers, qwen-vl-utils and autoawq are installed.")

    videos = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))
    print(f"Found {len(videos)} videos.")
    results = {}
    for path in videos:
        name = os.path.basename(path)
        print(f"Evaluating {name} ...")
        try:
            reply = evaluate_video(model, processor, path)
        except Exception as e:  # noqa: BLE001
            print(f"  failed: {e}")
            continue
        score = parse_score(reply)
        results[name] = {"score": score, "critique": reply}
        print(f"  score: {score if score is not None else 'unreadable'}")

        if args.sort and score is not None:
            dest = os.path.join(args.video_dir, "good_videos" if score >= args.threshold else "bad_videos")
            os.makedirs(dest, exist_ok=True)
            os.rename(path, os.path.join(dest, name))

    out_file = os.path.join(args.video_dir, "evaluation_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Done. Results saved to {out_file}")


if __name__ == "__main__":
    main()
