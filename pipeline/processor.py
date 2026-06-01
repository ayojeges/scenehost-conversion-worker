import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from pipeline.storage import download_url, upload_file


WORKDIR = Path(os.environ.get("SCENEHOST_WORKDIR", "/workspace/scenehost")) / "tmp"
GAUSSIAN_SPLATTING_DIR = Path(os.environ.get("SCENEHOST_GAUSSIAN_SPLATTING_DIR", "/opt/gaussian-splatting"))


def process_scene(payload, progress):
    scene_dir = WORKDIR / payload["sceneId"] / payload["jobId"]
    input_dir = scene_dir / "inputs"
    output_dir = scene_dir / "outputs"
    source_dir = scene_dir / "source"
    output_dir.mkdir(parents=True, exist_ok=True)

    progress({"stage": "preparing", "sceneId": payload["sceneId"], "jobId": payload["jobId"], "progress": 5})

    if payload["inputType"] == "dry_run":
        return dry_run(payload, output_dir, progress)

    downloaded = [download_url(url, input_dir) for url in payload["inputUrls"]]
    progress({"stage": "downloaded", "fileCount": len(downloaded), "progress": 12})

    if payload["inputType"] == "splat":
        return process_raw_splat(payload, downloaded, output_dir, progress)

    if payload["inputType"] == "video":
        return process_video(payload, downloaded, source_dir, output_dir, progress)

    if payload["inputType"] == "images":
        return process_images(payload, downloaded, source_dir, output_dir, progress)

    raise ValueError(f"Unsupported inputType: {payload['inputType']}")


def dry_run(payload, output_dir, progress):
    progress({"stage": "dry_run", "progress": 50})
    metadata_path = write_metadata(output_dir, {
        "sceneId": payload["sceneId"],
        "jobId": payload["jobId"],
        "mode": "dry_run",
        "message": "SceneHost worker contract is reachable.",
    })

    metadata_url = upload_file(metadata_path, f"{payload['outputPrefix']}/metadata.json", "application/json")
    progress({"stage": "completed", "progress": 100})
    return {
        "status": "completed",
        "sceneId": payload["sceneId"],
        "jobId": payload["jobId"],
        "inputType": payload["inputType"],
        "outputs": {"metadataUrl": metadata_url},
        "metrics": {"frameCount": 0, "outputBytes": metadata_path.stat().st_size},
    }


def process_raw_splat(payload, downloaded, output_dir, progress):
    progress({"stage": "optimizing_splat", "progress": 45})
    source = downloaded[0]
    suffix = source.suffix.lower() or ".ply"
    output_name = f"scene{suffix}"
    output_path = output_dir / output_name
    shutil.copyfile(source, output_path)

    compressed_path = optimize_splat(output_path, output_dir, progress)
    metadata_path = write_metadata(output_dir, {
        "sceneId": payload["sceneId"],
        "jobId": payload["jobId"],
        "mode": "raw_splat_passthrough",
        "sourceName": source.name,
        "optimized": compressed_path.name,
    })

    splat_url = upload_file(output_path, f"{payload['outputPrefix']}/{output_path.name}")
    compressed_url = upload_file(compressed_path, f"{payload['outputPrefix']}/{compressed_path.name}")
    metadata_url = upload_file(metadata_path, f"{payload['outputPrefix']}/metadata.json", "application/json")
    progress({"stage": "completed", "progress": 100})
    return {
        "status": "completed",
        "sceneId": payload["sceneId"],
        "jobId": payload["jobId"],
        "inputType": payload["inputType"],
        "outputs": {
            "splatUrl": splat_url,
            "compressedSplatUrl": compressed_url,
            "metadataUrl": metadata_url,
        },
        "metrics": {
            "inputBytes": source.stat().st_size,
            "outputBytes": compressed_path.stat().st_size,
        },
    }


def process_video(payload, downloaded, source_dir, output_dir, progress):
    video_path = downloaded[0]
    extracted_dir = source_dir / "_extracted_frames"
    images_dir = source_dir / "images"
    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)
    if images_dir.exists():
        shutil.rmtree(images_dir)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    options = payload.get("options") or {}
    frame_fps = float(options.get("frameFps") or os.environ.get("SCENEHOST_FRAME_FPS", "1"))
    max_frames = int(options.get("maxFrames") or os.environ.get("SCENEHOST_MAX_FRAMES", "240"))
    max_width = int(options.get("maxWidth") or os.environ.get("SCENEHOST_MAX_IMAGE_WIDTH", "1600"))

    progress({"stage": "extracting_frames", "progress": 20})
    run_command([
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={frame_fps},scale='min({max_width},iw)':-2",
        "-frames:v",
        str(max_frames),
        str(extracted_dir / "frame_%05d.jpg"),
    ])

    extracted_count = len(list(extracted_dir.glob("*.jpg")))
    if extracted_count < 8:
        raise RuntimeError(f"Not enough frames extracted for reconstruction: {extracted_count}")

    curation = curate_video_frames(extracted_dir, images_dir, options)
    frame_count = curation["selectedFrameCount"]
    min_frames = int((options.get("qualityGate") or {}).get("minFrames") or 80)
    if frame_count < max(8, min(80, min_frames)):
        raise RuntimeError(
            f"SceneHost could only find {frame_count} usable overlapping frames in this video. "
            "Record one slower continuous room path, avoid close wall shots, and keep each room in view longer."
        )

    frames = sorted(images_dir.glob("*.jpg"))
    preview_path = make_preview_from_image(frames[0], output_dir)
    return reconstruct_and_upload(payload, source_dir, output_dir, preview_path, progress, {
        "frameCount": frame_count,
        "extractedFrameCount": extracted_count,
        "inputBytes": video_path.stat().st_size,
        "frameCuration": curation,
    })


def process_images(payload, downloaded, source_dir, output_dir, progress):
    images_dir = source_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    progress({"stage": "normalizing_images", "progress": 20})
    normalized = []
    for index, image_path in enumerate(downloaded, start=1):
        suffix = image_path.suffix.lower()
        target = images_dir / f"image_{index:05d}{suffix if suffix in {'.jpg', '.jpeg', '.png'} else '.jpg'}"
        if suffix in {".jpg", ".jpeg", ".png"}:
            shutil.copyfile(image_path, target)
        else:
            run_command(["ffmpeg", "-y", "-i", str(image_path), str(target)])
        normalized.append(target)

    if len(normalized) < 8:
        raise RuntimeError(f"At least 8 images are required for reconstruction; received {len(normalized)}")

    preview_path = make_preview_from_image(normalized[0], output_dir)
    return reconstruct_and_upload(payload, source_dir, output_dir, preview_path, progress, {"imageCount": len(normalized)})


def curate_video_frames(extracted_dir, images_dir, options):
    """Select a coherent, high-information segment for 3DGS.

    Property walkthrough videos often include disconnected chapters: exterior,
    rooms, stairs, quick close-ups, and upstairs. COLMAP expects a continuous
    overlapping camera path, so a single reconstruction over every frame can
    smear unrelated spaces together. This step chooses the strongest contiguous
    segment and drops frames that are blurry, mostly blank wall, mostly sky/grass,
    or severe exposure outliers.
    """
    try:
        import cv2
        import numpy as np
    except Exception as error:
        # Worker images install opencv-python-headless, but if a custom image is
        # missing it, keep the job alive with a bounded sequential fallback.
        frames = sorted(extracted_dir.glob("*.jpg"))
        selected = frames[:int(options.get("maxFrames") or len(frames))]
        for index, frame in enumerate(selected, start=1):
            shutil.copyfile(frame, images_dir / f"frame_{index:05d}.jpg")
        return {
            "strategy": "sequential_fallback",
            "reason": f"OpenCV unavailable: {error}",
            "selectedFrameCount": len(selected),
            "extractedFrameCount": len(frames),
            "droppedFrameCount": max(0, len(frames) - len(selected)),
        }

    focus = str(options.get("reconstructionFocus") or os.environ.get("SCENEHOST_RECONSTRUCTION_FOCUS", "interior")).lower()
    min_segment_frames = int(options.get("minSegmentFrames") or os.environ.get("SCENEHOST_MIN_SEGMENT_FRAMES", "72"))
    max_selected = int(options.get("maxSelectedFrames") or options.get("maxFrames") or os.environ.get("SCENEHOST_MAX_SELECTED_FRAMES", "720"))
    frames = sorted(extracted_dir.glob("*.jpg"))
    records = []
    previous_hist = None

    for index, frame_path in enumerate(frames):
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        small = cv2.resize(image, (160, 90), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        brightness = float(gray.mean())
        contrast = float(gray.std())
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        edges = float((cv2.Canny(gray, 80, 160) > 0).mean())
        green_ratio = float((((hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 90) & (hsv[:, :, 1] > 45))).mean())
        sky_ratio = float((((hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 125) & (hsv[:, :, 1] > 25) & (hsv[:, :, 2] > 120))).mean())
        blank_ratio = float(((gray > 225) | (gray < 18)).mean())
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        scene_delta = 0.0 if previous_hist is None else float(cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        previous_hist = hist

        exposure_score = max(0.0, 1.0 - abs(brightness - 118.0) / 118.0)
        blur_score = min(1.0, blur / 90.0)
        contrast_score = min(1.0, contrast / 55.0)
        edge_score = min(1.0, edges / 0.09)
        exterior_penalty = max(green_ratio, sky_ratio * 0.8) if focus == "interior" else 0.0
        usable = (
            blur >= 12
            and 28 <= brightness <= 225
            and contrast >= 12
            and blank_ratio <= 0.62
            and not (focus == "interior" and green_ratio > 0.42)
            and not (focus == "interior" and sky_ratio > 0.35)
        )
        score = max(0.0, (
            blur_score * 0.24
            + contrast_score * 0.24
            + edge_score * 0.22
            + exposure_score * 0.18
            + (1.0 - blank_ratio) * 0.12
            - exterior_penalty * 0.28
        ))

        records.append({
            "path": frame_path,
            "index": index,
            "usable": usable,
            "score": score,
            "scene_delta": scene_delta,
            "greenRatio": green_ratio,
            "skyRatio": sky_ratio,
            "blur": blur,
            "brightness": brightness,
            "contrast": contrast,
        })

    if not records:
        raise RuntimeError("Frame curation failed: no readable frames were extracted.")

    segments = []
    current = []
    for record in records:
        hard_cut = record["scene_delta"] > 0.62
        if hard_cut and len(current) >= min_segment_frames:
            segments.append(current)
            current = []
        current.append(record)
    if current:
        segments.append(current)

    candidates = []
    for segment in segments:
        usable = [record for record in segment if record["usable"]]
        if len(usable) < max(24, min_segment_frames // 2):
            continue
        mean_score = sum(record["score"] for record in usable) / len(usable)
        mean_green = sum(record["greenRatio"] for record in usable) / len(usable)
        mean_sky = sum(record["skyRatio"] for record in usable) / len(usable)
        length_bonus = min(1.0, len(usable) / max(1, min_segment_frames * 2))
        focus_bonus = 1.0 - max(mean_green, mean_sky * 0.8) if focus == "interior" else 1.0
        candidates.append({
            "segment": segment,
            "usable": usable,
            "rank": mean_score * 0.7 + length_bonus * 0.2 + focus_bonus * 0.1,
            "meanScore": mean_score,
            "meanGreenRatio": mean_green,
            "meanSkyRatio": mean_sky,
        })

    if candidates:
        chosen = max(candidates, key=lambda item: item["rank"])
        selected = chosen["usable"]
        strategy = "best_coherent_segment"
    else:
        # Last-resort fallback: take the best frames globally but preserve time order.
        selected = sorted(records, key=lambda record: record["score"], reverse=True)[:max(24, min(len(records), max_selected))]
        selected = sorted(selected, key=lambda record: record["index"])
        chosen = {
            "segment": selected,
            "usable": selected,
            "rank": 0,
            "meanScore": sum(record["score"] for record in selected) / max(1, len(selected)),
            "meanGreenRatio": sum(record["greenRatio"] for record in selected) / max(1, len(selected)),
            "meanSkyRatio": sum(record["skyRatio"] for record in selected) / max(1, len(selected)),
        }
        strategy = "global_best_frames_fallback"

    if len(selected) > max_selected:
        step = len(selected) / max_selected
        selected = [selected[int(i * step)] for i in range(max_selected)]

    for index, record in enumerate(selected, start=1):
        shutil.copyfile(record["path"], images_dir / f"frame_{index:05d}.jpg")

    chosen_indices = [record["index"] for record in selected]
    return {
        "strategy": strategy,
        "focus": focus,
        "selectedFrameCount": len(selected),
        "extractedFrameCount": len(records),
        "droppedFrameCount": max(0, len(records) - len(selected)),
        "segmentStartFrame": min(chosen_indices) + 1,
        "segmentEndFrame": max(chosen_indices) + 1,
        "meanFrameScore": round(float(chosen["meanScore"]), 3),
        "meanGreenRatio": round(float(chosen["meanGreenRatio"]), 3),
        "meanSkyRatio": round(float(chosen["meanSkyRatio"]), 3),
    }


def reconstruct_and_upload(payload, source_dir, output_dir, preview_path, progress, metrics):
    started = time.time()
    options = payload.get("options") or {}
    preflight_only = str(options.get("preflightOnly") or os.environ.get("SCENEHOST_PREFLIGHT_ONLY", "")).lower() in {"1", "true", "yes"}
    cost_guard = options.get("costGuard") or {}

    if preflight_only:
        metadata_path = write_metadata(output_dir, {
            "sceneId": payload["sceneId"],
            "jobId": payload["jobId"],
            "mode": "preflight_only",
            "message": "Media prepared successfully; training skipped by SCENEHOST_PREFLIGHT_ONLY.",
            **metrics,
        })
        metadata_url = upload_file(metadata_path, f"{payload['outputPrefix']}/metadata.json", "application/json")
        preview_url = upload_file(preview_path, f"{payload['outputPrefix']}/preview.jpg", "image/jpeg")
        return {
            "status": "needs_training_pipeline",
            "sceneId": payload["sceneId"],
            "jobId": payload["jobId"],
            "inputType": payload["inputType"],
            "outputs": {"metadataUrl": metadata_url, "previewImageUrl": preview_url},
            "metrics": metrics,
        }

    progress({"stage": "training_3dgs", "progress": 35})
    ply_path = train_gaussian_splat(source_dir, output_dir, int(options.get("iterations") or os.environ.get("SCENEHOST_3DGS_ITERATIONS", "7000")))
    registered_images = count_registered_images(source_dir / "sparse" / "0")
    source_count = int(metrics.get("frameCount") or metrics.get("imageCount") or 0)
    registered_ratio = round(registered_images / source_count, 3) if source_count else 0
    min_registered_ratio = float(cost_guard.get("stopIfRegisteredFrameRatioBelow") or options.get("qualityGate", {}).get("minRegisteredFrameRatio") or 0)
    if min_registered_ratio and registered_ratio and registered_ratio < min_registered_ratio:
        raise RuntimeError(
            f"Reconstruction quality too low: only {registered_images}/{source_count} frames aligned "
            f"({registered_ratio}). Capture a slower walkthrough with more overlap."
        )

    progress({"stage": "optimizing_splat", "progress": 82})
    compressed_path = optimize_splat(ply_path, output_dir, progress)
    processing_seconds = round(time.time() - started, 2)
    splat_count = count_ply_vertices(ply_path)
    estimated_cost = estimate_gpu_cost(processing_seconds, options)

    metadata = {
        "sceneId": payload["sceneId"],
        "jobId": payload["jobId"],
        "mode": "video_image_3dgs",
        "sourceType": payload["inputType"],
        "trainingSeconds": processing_seconds,
        "ply": ply_path.name,
        "optimized": compressed_path.name,
        "registeredImages": registered_images,
        "registeredFrameRatio": registered_ratio,
        "splatCount": splat_count,
        "estimatedCostUsd": estimated_cost,
        **metrics,
    }
    metadata_path = write_metadata(output_dir, metadata)

    splat_url = upload_file(ply_path, f"{payload['outputPrefix']}/{ply_path.name}")
    compressed_url = upload_file(compressed_path, f"{payload['outputPrefix']}/{compressed_path.name}")
    preview_url = upload_file(preview_path, f"{payload['outputPrefix']}/preview.jpg", "image/jpeg")
    metadata_url = upload_file(metadata_path, f"{payload['outputPrefix']}/metadata.json", "application/json")

    progress({"stage": "completed", "progress": 100})
    return {
        "status": "completed",
        "sceneId": payload["sceneId"],
        "jobId": payload["jobId"],
        "inputType": payload["inputType"],
        "outputs": {
            "splatUrl": splat_url,
            "compressedSplatUrl": compressed_url,
            "previewImageUrl": preview_url,
            "metadataUrl": metadata_url,
        },
        "metrics": {
            **metrics,
            "processingSeconds": processing_seconds,
            "gpuSeconds": processing_seconds,
            "estimatedCostUsd": estimated_cost,
            "registeredImages": registered_images,
            "registeredFrameRatio": registered_ratio,
            "splatCount": splat_count,
            "outputBytes": compressed_path.stat().st_size,
            "rawOutputBytes": ply_path.stat().st_size,
            "optimizedBytes": compressed_path.stat().st_size,
            "compressionSavingsPercent": compression_savings_percent(ply_path, compressed_path),
        },
    }


def count_registered_images(sparse_zero):
    images_txt = sparse_zero / "images.txt"
    images_bin = sparse_zero / "images.bin"
    if not images_txt.exists() and images_bin.exists():
        text_dir = sparse_zero.parent / "_metric_text_model"
        if text_dir.exists():
            shutil.rmtree(text_dir)
        text_dir.mkdir(parents=True, exist_ok=True)
        run_command([
            "colmap",
            "model_converter",
            "--input_path",
            str(sparse_zero),
            "--output_path",
            str(text_dir),
            "--output_type",
            "TXT",
        ])
        images_txt = text_dir / "images.txt"

    if not images_txt.exists():
      return 0

    count = 0
    for line in images_txt.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 10 and "." in parts[-1]:
            count += 1
    shutil.rmtree(sparse_zero.parent / "_metric_text_model", ignore_errors=True)
    return count


def count_ply_vertices(ply_path):
    try:
        for line in ply_path.read_text(encoding="utf-8", errors="ignore").splitlines()[:80]:
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line.strip() == "end_header":
                break
    except Exception:
        return 0
    return 0


def estimate_gpu_cost(processing_seconds, options):
    provided = options.get("estimatedCostUsd")
    if isinstance(provided, (int, float)) and provided > 0:
        return round(float(provided), 2)
    return round(float(processing_seconds) * 0.00075, 2)


def compression_savings_percent(raw_path, optimized_path):
    raw_size = raw_path.stat().st_size
    optimized_size = optimized_path.stat().st_size
    if raw_size <= 0:
        return 0
    return round(max(0, (1 - optimized_size / raw_size) * 100), 2)


def train_gaussian_splat(source_dir, output_dir, iterations):
    custom_command = os.environ.get("SCENEHOST_TRAIN_COMMAND", "").strip()
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    if custom_command:
        command = custom_command.format(source=str(source_dir), model=str(model_dir), iterations=iterations)
        run_command(command, shell=True)
    else:
        if not GAUSSIAN_SPLATTING_DIR.exists():
            raise RuntimeError(
                f"3DGS backend not found at {GAUSSIAN_SPLATTING_DIR}. "
                "Set SCENEHOST_GAUSSIAN_SPLATTING_DIR or SCENEHOST_TRAIN_COMMAND."
            )

        train_py = GAUSSIAN_SPLATTING_DIR / "train.py"
        use_graphdeco_convert = os.environ.get("SCENEHOST_USE_GRAPHDECO_CONVERT", "").lower() in {"1", "true", "yes"}
        if use_graphdeco_convert:
            convert_py = GAUSSIAN_SPLATTING_DIR / "convert.py"
            run_command([
                "python3",
                str(convert_py),
                "-s",
                str(source_dir),
                "--resize",
                "--no_gpu",
                "--camera",
                "PINHOLE",
            ])

        ensure_colmap_scene(source_dir)
        normalize_colmap_cameras(source_dir / "sparse" / "0")
        run_command([
            "python3",
            str(train_py),
            "-s",
            str(source_dir),
            "-m",
            str(model_dir),
            "--iterations",
            str(iterations),
            "--quiet",
        ])

    candidates = sorted(model_dir.glob("point_cloud/iteration_*/point_cloud.ply"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        candidates = sorted(output_dir.glob("**/*.ply"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError("3DGS training completed but no point_cloud.ply was produced.")

    final_ply = output_dir / "scene.ply"
    shutil.copyfile(candidates[0], final_ply)
    return final_ply


def ensure_colmap_scene(source_dir):
    sparse_zero = source_dir / "sparse" / "0"
    if sparse_zero.exists():
        return

    distorted_sparse_zero = source_dir / "distorted" / "sparse" / "0"
    if distorted_sparse_zero.exists():
        sparse_zero.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(distorted_sparse_zero, sparse_zero, dirs_exist_ok=True)
        return

    images_dir = source_dir / "images"
    if not images_dir.exists():
        raise RuntimeError(f"COLMAP scene preparation failed: {images_dir} does not exist.")

    database_path = source_dir / "colmap.db"
    sparse_dir = source_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    run_command([
        "colmap",
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(images_dir),
        "--ImageReader.single_camera",
        "1",
        "--SiftExtraction.use_gpu",
        "0",
    ])
    run_command([
        "colmap",
        "exhaustive_matcher",
        "--database_path",
        str(database_path),
        "--SiftMatching.use_gpu",
        "0",
    ])
    mapper_result = run_command([
        "colmap",
        "mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(images_dir),
        "--output_path",
        str(sparse_dir),
    ], allow_failure=True)

    if not sparse_zero.exists():
        available = ", ".join(str(path.relative_to(source_dir)) for path in sparse_dir.glob("**/*") if path.is_file())[:500]
        if mapper_result.returncode != 0:
            raise RuntimeError(
                "COLMAP scene preparation failed before producing a usable sparse model.\n"
                f"STDOUT:\n{mapper_result.stdout[-2000:]}\nSTDERR:\n{mapper_result.stderr[-2000:]}"
            )
        raise RuntimeError(f"COLMAP scene preparation failed: sparse/0 was not produced. Available files: {available}")


def normalize_colmap_cameras(sparse_zero):
    cameras_bin = sparse_zero / "cameras.bin"
    if not cameras_bin.exists():
        return

    text_dir = sparse_zero.parent / "_text_model"
    if text_dir.exists():
        shutil.rmtree(text_dir)
    text_dir.mkdir(parents=True, exist_ok=True)

    run_command([
        "colmap",
        "model_converter",
        "--input_path",
        str(sparse_zero),
        "--output_path",
        str(text_dir),
        "--output_type",
        "TXT",
    ])

    cameras_txt = text_dir / "cameras.txt"
    normalized = []
    changed = False
    for line in cameras_txt.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            normalized.append(line)
            continue

        parts = line.split()
        model = parts[1]
        if model in {"PINHOLE", "SIMPLE_PINHOLE"}:
            normalized.append(line)
            continue

        if len(parts) < 8:
            raise RuntimeError(f"Unexpected COLMAP camera line: {line}")

        camera_id, _, width, height = parts[:4]
        fx, fy, cx, cy = parts[4:8]
        normalized.append(" ".join([camera_id, "PINHOLE", width, height, fx, fy, cx, cy]))
        changed = True

    if not changed:
        shutil.rmtree(text_dir, ignore_errors=True)
        return

    cameras_txt.write_text("\n".join(normalized) + "\n", encoding="utf-8")
    run_command([
        "colmap",
        "model_converter",
        "--input_path",
        str(text_dir),
        "--output_path",
        str(sparse_zero),
        "--output_type",
        "BIN",
    ])
    shutil.rmtree(text_dir, ignore_errors=True)


def optimize_splat(input_path, output_dir, progress):
    command_template = os.environ.get("SCENEHOST_SPLAT_TRANSFORM_CMD", "").strip()
    output_path = output_dir / "scene.compressed.ply"
    if command_template:
        command = command_template.format(input=str(input_path), output=str(output_path))
        run_command(command, shell=True)
        if output_path.exists():
            return output_path

    if shutil.which("splat-transform"):
        run_command([
            "splat-transform",
            str(input_path),
            "-N",
            "-H",
            "0",
            "-M",
            str(output_path),
            "-w",
            "--no-tty",
            "-q",
        ])
        if output_path.exists() and output_path.stat().st_size > 0:
            return output_path

    if input_path.name != "scene.ply":
        fallback = output_dir / "scene.ply"
        shutil.copyfile(input_path, fallback)
        return fallback

    return input_path


def make_preview_from_image(image_path, output_dir):
    preview_path = output_dir / "preview.jpg"
    run_command([
        "ffmpeg",
        "-y",
        "-i",
        str(image_path),
        "-frames:v",
        "1",
        "-vf",
        "scale='min(1280,iw)':-2",
        str(preview_path),
    ])
    return preview_path


def write_metadata(output_dir, data):
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return metadata_path


def run_command(command, shell=False, allow_failure=False):
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("XDG_RUNTIME_DIR", "/tmp/runtime-root")
    Path(env["XDG_RUNTIME_DIR"]).mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, capture_output=True, text=True, check=False, shell=shell, env=env)
    if result.returncode != 0 and not allow_failure:
        rendered = command if isinstance(command, str) else " ".join(command)
        raise RuntimeError(
            f"Command failed: {rendered}\nSTDOUT:\n{result.stdout[-2000:]}\nSTDERR:\n{result.stderr[-2000:]}"
        )
    return result
