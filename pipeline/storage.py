import os
from pathlib import Path
from urllib.parse import urlparse

import boto3
import requests
from botocore.config import Config


def download_url(url, destination_dir):
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(url)
    filename = Path(parsed.path).name or "input.bin"
    destination = destination_dir / filename

    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

    return destination


def r2_client():
    endpoint = os.environ.get("R2_ENDPOINT")
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if not endpoint and account_id:
        endpoint = f"https://{account_id}.r2.cloudflarestorage.com"

    required = {
        "R2 endpoint": endpoint,
        "R2_ACCESS_KEY_ID": os.environ.get("R2_ACCESS_KEY_ID"),
        "R2_SECRET_ACCESS_KEY": os.environ.get("R2_SECRET_ACCESS_KEY"),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(f"Missing storage environment variables: {', '.join(missing)}")

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )


def upload_file(local_path, object_key, content_type="application/octet-stream"):
    bucket = os.environ.get("R2_BUCKET", "scenehost-assets")
    client = r2_client()
    client.upload_file(
        str(local_path),
        bucket,
        object_key,
        ExtraArgs={"ContentType": content_type},
    )

    public_base_url = os.environ.get("R2_PUBLIC_BASE_URL", "").rstrip("/")
    if public_base_url:
        return f"{public_base_url}/{object_key}"
    return f"r2://{bucket}/{object_key}"
