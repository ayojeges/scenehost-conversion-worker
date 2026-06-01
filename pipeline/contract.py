REQUIRED_FIELDS = ("jobId", "sceneId", "inputType", "inputUrls", "outputPrefix")
SUPPORTED_INPUT_TYPES = {"dry_run", "splat", "video", "images"}


def validate_job_input(job_input):
    if not isinstance(job_input, dict):
        raise ValueError("RunPod job input must be an object.")

    missing = [field for field in REQUIRED_FIELDS if field not in job_input]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    input_type = job_input["inputType"]
    if input_type not in SUPPORTED_INPUT_TYPES:
        raise ValueError(f"Unsupported inputType: {input_type}")

    if not isinstance(job_input["inputUrls"], list):
        raise ValueError("inputUrls must be a list.")

    if input_type != "dry_run" and not job_input["inputUrls"]:
        raise ValueError("inputUrls must contain at least one URL for non-dry-run jobs.")

    return {
        "jobId": str(job_input["jobId"]),
        "sceneId": str(job_input["sceneId"]),
        "inputType": input_type,
        "inputUrls": job_input["inputUrls"],
        "outputPrefix": str(job_input["outputPrefix"]).strip("/"),
        "webhookUrl": job_input.get("webhookUrl") or "",
        "webhookAuth": (job_input.get("options") or {}).get("webhookAuth") or job_input.get("webhookAuth") or {},
        "options": job_input.get("options") or {},
    }
