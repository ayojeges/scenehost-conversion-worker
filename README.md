# SceneHost RunPod Worker

This is the first deployable worker contract for SceneHost conversion jobs.

It supports these stages:

- `dry_run`: verifies RunPod -> worker -> R2 -> app/webhook contract.
- `splat`: downloads an existing splat and uploads a web-hosted output.
- `video`: extracts frames with FFmpeg, runs the configured 3DGS backend, uploads `.ply`, optimized splat output, preview image, and metadata.
- `images`: normalizes an image set, runs the configured 3DGS backend, uploads `.ply`, optimized splat output, preview image, and metadata.

The default training backend expects the Graphdeco Gaussian Splatting repository at `/opt/gaussian-splatting`.
You can also provide a backend command with:

```bash
SCENEHOST_TRAIN_COMMAND="python3 /opt/my-trainer/train.py --source {source} --model {model} --iterations {iterations}"
```

For infrastructure-only testing, set:

```bash
SCENEHOST_PREFLIGHT_ONLY=1
```

This keeps video/image media preparation active while skipping GPU training.

## Quality and cost tuning

The app sends a centralized tuning contract in `input.options`:

- `quality`: `preview` or `standard`
- `frameFps`, `maxFrames`, `maxWidth`, `iterations`
- `qualityGate.minFrames`, `minQualityScore`, `minViewerQualityScore`, `minRegisteredFrameRatio`, `minCoverageScore`, `minSplatCount`
- `costGuard.targetGpuSeconds`, `estimatedCostUsd`, `stopIfNoCameraAlignmentAfterSeconds`, `stopIfRegisteredFrameRatioBelow`

The worker reports `frameCount` or `imageCount`, `registeredImages`, `registeredFrameRatio`, `splatCount`, `gpuSeconds`, `estimatedCostUsd`, `rawOutputBytes`, `optimizedBytes`, and `compressionSavingsPercent`.

If frame alignment falls below the app threshold, the worker fails with a capture-quality message so SceneHost can keep weak reconstructions out of the public viewer.

RunPod workers receive jobs as:

```json
{
  "input": {
    "jobId": "conversion-job-id",
    "sceneId": "scene-id",
    "inputType": "dry_run",
    "inputUrls": [],
    "outputPrefix": "scenes/scene-id/outputs",
    "webhookUrl": "https://scenehost.com/api/webhooks/runpod"
  }
}
```

## Local Smoke Test

```powershell
python -m venv .venv
.\\.venv\\Scripts\\pip install -r requirements.txt
.\\.venv\\Scripts\\python handler.py --test_input (Get-Content .\\test_input.json -Raw)
```

## Build Image

Docker is required to build the worker image.

```powershell
docker build -t scenehost-conversion:latest .
```

Push the image to a registry, then use `scripts/create-runpod-template.cjs` and `scripts/create-runpod-endpoint.cjs` from the parent `scenehost` folder.

## Output Contract

Successful video/image jobs return:

```json
{
  "status": "completed",
  "outputs": {
    "splatUrl": "r2 or public URL to scene.ply",
    "compressedSplatUrl": "r2 or public URL to scene.compressed.ply or scene.ply",
    "previewImageUrl": "r2 or public URL to preview.jpg",
    "metadataUrl": "r2 or public URL to metadata.json"
  }
}
```
