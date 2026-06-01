import hashlib
import hmac
import json
import time

import requests


def post_webhook(webhook_url, payload, webhook_auth=None):
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    headers = {"Content-Type": "application/json"}
    secret = (webhook_auth or {}).get("secret")
    if secret:
        timestamp = str(int(time.time() * 1000))
        signature = hmac.new(secret.encode("utf-8"), f"{timestamp}.{body}".encode("utf-8"), hashlib.sha256).hexdigest()
        headers[(webhook_auth or {}).get("headerTimestamp") or "x-scenehost-timestamp"] = timestamp
        headers[(webhook_auth or {}).get("headerSignature") or "x-scenehost-signature"] = signature

    response = requests.post(webhook_url, data=body, headers=headers, timeout=30)
    response.raise_for_status()
    return response
