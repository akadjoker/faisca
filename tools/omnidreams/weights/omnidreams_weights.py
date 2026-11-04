import hashlib
import os
import time

from huggingface_hub import hf_hub_download
from kaggle_secrets import UserSecretsClient

token = UserSecretsClient().get_secret("HF_TOKEN")
started = time.time()
path = hf_hub_download(
    repo_id="nvidia/omni-dreams-models",
    filename="single_view/2b_res720p_30fps_i2v_hdmap_distilled.pt",
    local_dir="/kaggle/working",
    token=token,
)
token = None
print("downloaded", path, os.path.getsize(path), "bytes in %.0f s" % (time.time() - started))
digest = hashlib.sha256()
with open(path, "rb") as handle:
    for block in iter(lambda: handle.read(1 << 24), b""):
        digest.update(block)
with open("/kaggle/working/SHA256SUMS", "w") as handle:
    handle.write(digest.hexdigest() + "  single_view/2b_res720p_30fps_i2v_hdmap_distilled.pt\n")
print("sha256", digest.hexdigest())
for name in [".cache"]:
    os.system(f"rm -rf /kaggle/working/{name}")
