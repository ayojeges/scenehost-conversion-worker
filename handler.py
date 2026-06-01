import json
import os
import time
import traceback
from pathlib import Path

import runpod

from pipeline.contract import validate_job_input
from pipeline.processor import process_scene
from pipeline.webhook import post_webhook

WORKER_VERSION = os.environ.get("SCENEHOST_WORKER_VERSION", "2026-05-31-reconstruction-v2")


def handler(job):
    started_at = time.time()
    runpod_job_id = job.get("id")
    job_input = job.get("input") or {}

    try:
        payload = validate_job_input(job_input)
        runpod.serverless.progress_update(job, {
            "stage": "accepted",
            "sceneId": payload["sceneId"],
            "jobId": payload["jobId"],
        })

        result = process_scene(payload, progress=lambda update: runpod.serverless.progress_update(job, update))
        result["runpodJobId"] = runpod_job_id
        result["workerVersion"] = WORKER_VERSION
        result["processingSeconds"] = round(time.time() - started_at, 2)

        if payload.get("webhookUrl"):
            post_webhook(payload["webhookUrl"], result, payload.get("webhookAuth"))

        return result
    except Exception as exc:
        error_result = {
            "status": "failed",
            "runpodJobId": runpod_job_id,
            "jobId": job_input.get("jobId"),
            "sceneId": job_input.get("sceneId"),
            "workerVersion": WORKER_VERSION,
            "error": str(exc),
            "trace": traceback.format_exc().splitlines()[-8:],
            "processingSeconds": round(time.time() - started_at, 2),
        }

        webhook_url = job_input.get("webhookUrl")
        if webhook_url:
            try:
                post_webhook(webhook_url, error_result, (job_input.get("options") or {}).get("webhookAuth") or job_input.get("webhookAuth") or {})
            except Exception:
                pass

        return error_result


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
