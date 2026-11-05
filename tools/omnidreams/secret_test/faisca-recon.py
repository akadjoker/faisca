import os
import traceback

print("KAGGLE env names:", sorted(name for name in os.environ if "KAGGLE" in name or "SECRET" in name))
try:
    from kaggle_secrets import UserSecretsClient
    client = UserSecretsClient()
    value = client.get_secret("HF_TOKEN")
    print("HF_TOKEN: present, length", len(value), "prefix ok", value.startswith("hf_"))
except Exception:
    lines = traceback.format_exc().splitlines()
    print("\n".join(line for line in lines if "hf_" not in line))

import urllib.request
try:
    token = UserSecretsClient().get_secret("HF_TOKEN")
    url = "https://huggingface.co/nvidia/omni-dreams-models/resolve/main/single_view/2b_res720p_30fps_i2v_hdmap_distilled.pt"
    request = urllib.request.Request(url, method="HEAD", headers={"Authorization": "Bearer " + token})
    response = urllib.request.urlopen(request, timeout=30)
    print("gated checkpoint HEAD:", response.status, "size", response.headers.get("Content-Length") or response.headers.get("X-Linked-Size"))
except Exception as error:
    print("gated checkpoint HEAD failed:", type(error).__name__, getattr(error, "code", ""))
