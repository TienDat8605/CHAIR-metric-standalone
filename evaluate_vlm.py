"""
Generate captions for the COCO val 2014 set using
`unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit` and score them with CHAIR.

Pipeline:
  1. Load the 4-bit model with unsloth FastModel.
  2. For every COCO val2014 image, prompt "Describe this image." with
     greedy decoding and append {"image_id": ..., "caption": ...} to
     a JSONL file (incremental, resume-aware).
  3. After all captions are written, rebuild chair.pkl from
     coco_annotations/ and run chair.py to obtain CHAIRs / CHAIRi / Recall.

Usage:
    python evaluate_vlm.py [--max-images 50]   # smoke test
    python evaluate_vlm.py                    # full val2014
    python evaluate_vlm.py --evaluate-only    # skip inference, re-score existing JSONL
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# Enable hf_transfer for fast HF downloads
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# Make CUDA 13.0 runtime libs (libnvJitLink.so.13, libnvrtc-builtins.so.13.0)
# discoverable when bundled inside the venv's nvidia/cu13/ directory. Required
# for bitsandbytes 4-bit kernels and triton kernel compilation on cu130.
_nvidia_lib = Path(__file__).parent / ".venv" / "lib" / "python3.12" / "site-packages" / "nvidia" / "cu13" / "lib"
if _nvidia_lib.is_dir():
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{_nvidia_lib}{os.pathsep}{cur}" if cur else str(_nvidia_lib)

from unsloth import FastModel
from transformers import AutoProcessor


DEFAULT_MODEL = "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit"
DEFAULT_PROMPT = "Describe this image."
DEFAULT_IMAGES_DIR = Path("coco/val2014")
DEFAULT_OUTPUT = Path("captions.jsonl")
DEFAULT_ANNOTATIONS = Path("coco_annotations")
DEFAULT_CACHE = Path("chair.pkl")
DEFAULT_RESULTS = Path("chair_results.json")
FILENAME_PREFIX = "COCO_val2014_"
FILENAME_SUFFIX = ".jpg"


def parse_image_id(path: Path) -> int:
    """COCO_val2014_000000391895.jpg -> 391895"""
    stem = path.stem  # 'COCO_val2014_000000391895'
    return int(stem.replace(FILENAME_PREFIX, ""))


def load_existing_ids(output_path: Path) -> set:
    ids = set()
    if not output_path.exists():
        return ids
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(int(json.loads(line)["image_id"]))
            except Exception:
                continue
    return ids


def build_message(prompt: str, pil_image: Image.Image) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": prompt},
        ],
    }


def caption_batch(model, processor, prompt: str, pil_images, max_new_tokens: int) -> list[str]:
    """Caption a batch of PIL images in a single forward pass."""
    messages_list = [[build_message(prompt, img)] for img in pil_images]
    inputs = processor.apply_chat_template(
        messages_list,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )

    in_ids = inputs["input_ids"]
    trimmed = [out[len(in_ids[i]):] for i, out in enumerate(generated_ids)]
    texts = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return [t.strip() for t in texts]


def run_inference(args, model, processor) -> int:
    images_dir: Path = args.images_dir
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not images_dir.is_dir():
        raise FileNotFoundError(f"images dir not found: {images_dir.resolve()}")

    all_images = sorted(images_dir.glob(f"{FILENAME_PREFIX}*{FILENAME_SUFFIX}"))
    if not all_images:
        raise FileNotFoundError(f"no images matching {FILENAME_PREFIX}*{FILENAME_SUFFIX} in {images_dir}")

    if args.max_images and args.max_images > 0:
        all_images = all_images[: args.max_images]

    done_ids = load_existing_ids(output_path)
    todo = [p for p in all_images if parse_image_id(p) not in done_ids]
    print(f"[info] {len(all_images)} images total, {len(done_ids)} already done, "
          f"{len(todo)} to process (batch_size={args.batch_size})")

    if not todo:
        return 0

    # Append-mode JSONL, flush every 50 entries
    flush_every = 50
    buffer: list[str] = []
    written = 0
    t0 = time.time()
    max_side = args.max_image_side
    bs = args.batch_size
    with open(output_path, "a", encoding="utf-8") as f:
        for start in tqdm(range(0, len(todo), bs), desc=f"captioning (bs={bs})"):
            batch_paths = todo[start:start + bs]
            batch_imgs = []
            batch_ids = []
            try:
                for img_path in batch_paths:
                    imid = parse_image_id(img_path)
                    pil = Image.open(img_path).convert("RGB")
                    if max(pil.size) > max_side:
                        pil.thumbnail((max_side, max_side), Image.LANCZOS)
                    batch_imgs.append(pil)
                    batch_ids.append(imid)
                captions = caption_batch(
                    model, processor, args.prompt, batch_imgs, args.max_new_tokens
                )
            except torch.cuda.OutOfMemoryError as e:
                print(f"\n[warn] CUDA OOM on batch starting at {batch_paths[0].name}: {e}. Skipping batch.")
                torch.cuda.empty_cache()
                captions = [""] * len(batch_paths)
                batch_ids = [parse_image_id(p) for p in batch_paths]
            except Exception as e:
                print(f"\n[warn] failed on batch starting at {batch_paths[0].name}: {e}. Skipping batch.")
                captions = [""] * len(batch_paths)
                batch_ids = [parse_image_id(p) for p in batch_paths]

            for imid, caption in zip(batch_ids, captions):
                buffer.append(json.dumps({"image_id": imid, "caption": caption},
                                         ensure_ascii=False))
                written += 1
            if written % flush_every < bs or len(buffer) >= flush_every:
                f.write("\n".join(buffer) + "\n")
                f.flush()
                buffer.clear()
            torch.cuda.empty_cache()

        if buffer:
            f.write("\n".join(buffer) + "\n")
            f.flush()

    elapsed = time.time() - t0
    rate = written / max(elapsed, 1e-6)
    print(f"[info] wrote {written} captions in {elapsed:.1f}s ({rate:.2f} img/s)")
    return written


def run_chair(args) -> int:
    annotations_dir: Path = args.annotations_dir
    cache_path: Path = args.cache
    results_path: Path = args.results
    output_path: Path = args.output

    if not annotations_dir.is_dir():
        raise FileNotFoundError(f"annotations dir not found: {annotations_dir.resolve()}")
    for fn in ("captions_train2014.json", "captions_val2014.json",
               "instances_train2014.json", "instances_val2014.json"):
        if not (annotations_dir / fn).exists():
            raise FileNotFoundError(f"missing annotation file: {annotations_dir / fn}")

    if not output_path.exists():
        raise FileNotFoundError(f"caption file not found: {output_path.resolve()}")

    # Force rebuild of chair.pkl (user opted in to "rebuild from coco_annotations/")
    if args.rebuild_cache and cache_path.exists():
        print(f"[info] deleting existing cache {cache_path} to force rebuild")
        cache_path.unlink()

    cmd = [
        sys.executable, "chair.py",
        "--cap_file", str(output_path),
        "--image_id_key", "image_id",
        "--caption_key", "caption",
        "--cache", str(cache_path),
        "--coco_path", str(annotations_dir),
        "--save_path", str(results_path),
    ]
    print(f"[info] running: {' '.join(cmd)}")
    rc = subprocess.call(cmd, cwd=Path(__file__).parent.resolve())
    return rc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--max-images", type=int, default=0,
                   help="Process at most N images (0 = all). Useful for smoke tests.")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4,
                   help="Number of images to caption per forward pass. Tune for your VRAM.")
    p.add_argument("--max-image-side", type=int, default=896,
                   help="Down-scale images so that max(width,height) <= this value.")
    p.add_argument("--evaluate-only", action="store_true",
                   help="Skip inference; just run CHAIR on the existing JSONL.")
    p.add_argument("--rebuild-cache", action="store_true",
                   help="Delete chair.pkl and rebuild from coco_annotations/.")
    p.add_argument("--max-seq-length", type=int, default=2048)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.evaluate_only:
        print(f"[info] loading model {args.model} (4-bit) via unsloth FastModel")
        model, tokenizer = FastModel.from_pretrained(
            model_name=args.model,
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,
        )
        # Switch on unsloth's inference kernels
        model = FastModel.for_inference(model)
        processor = AutoProcessor.from_pretrained(args.model)

        written = run_inference(args, model, processor)
        if written == 0 and load_existing_ids(args.output):
            print("[info] nothing new to caption; proceeding to CHAIR")
        elif written == 0:
            print("[error] no captions were written; aborting")
            return 1

        # Free GPU memory before invoking chair.py in a subprocess
        del model, processor
        torch.cuda.empty_cache()
    else:
        print("[info] --evaluate-only set; skipping inference")

    rc = run_chair(args)
    if rc == 0 and args.results.exists():
        try:
            with open(args.results, "r", encoding="utf-8") as f:
                overall = json.load(f).get("overall_metrics", {})
            print("\n=== CHAIR overall metrics ===")
            for k, v in overall.items():
                print(f"  {k:8s}: {v * 100:.2f}")
        except Exception as e:
            print(f"[warn] could not parse results: {e}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
