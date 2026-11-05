import os
import subprocess
import time
import urllib.request

OUT = "/kaggle/working"
LOG = os.path.join(OUT, "recon.txt")
log_file = open(LOG, "w")


def say(text):
    print(text, flush=True)
    log_file.write(text + "\n")
    log_file.flush()


def run(title, command, timeout=600):
    say(f"\n===== {title}\n$ {command}")
    started = time.time()
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
        output = (result.stdout + result.stderr).strip()
        lines = output.splitlines()
        if len(lines) > 80:
            output = "\n".join(lines[:20] + ["... (%d lines cut) ..." % (len(lines) - 70)] + lines[-50:])
        say(output)
        say(f"[exit {result.returncode}, {time.time() - started:.1f} s]")
        return result.returncode
    except subprocess.TimeoutExpired:
        say(f"[timeout after {timeout} s]")
        return -1


run("gpu", "nvidia-smi; nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv")
run("cpu", "nproc; lscpu | grep -E 'Model name|Thread|Core|Socket'; grep -o -w -E 'avx2|avx512f|avx512bw|avx512_vnni|avx512_bf16|avx_vnni|amx_int8|f16c|fma' /proc/cpuinfo | sort -u | tr '\\n' ' '")
run("memory", "free -g")
run("disk", "df -h /kaggle/working /kaggle/temp /tmp / 2>/dev/null")
run("os", "cat /etc/os-release | head -3; uname -r; python3 --version")
run("nvcc", "which nvcc; nvcc --version; ls -d /usr/local/cuda*")
run("torch", "python3 -c \"import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count()); "
    "[print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i), torch.cuda.get_device_properties(i).total_memory // 2**20, 'MiB') for i in range(torch.cuda.device_count())]\"")
run("vulkan", "ls /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>&1; ldconfig -p | grep -i -E 'vulkan|nvidia-glcore|GLX_nvidia' | head; which vulkaninfo")
run("cmake ninja", "cmake --version | head -1; ninja --version; gcc --version | head -1")

say("\n===== internet and hugging face")
try:
    started = time.time()
    urllib.request.urlopen("https://huggingface.co/api/models/nvidia/omni-dreams-models", timeout=30).read()
    say(f"huggingface reachable ({time.time() - started:.1f} s)")
except Exception as error:
    say(f"huggingface not reachable: {error!r}")

src = "/tmp/src"
os.makedirs(src, exist_ok=True)
run("clone flashdreams", f"git clone -q --depth 1 https://github.com/NVIDIA/flashdreams.git {src}/flashdreams && du -sh {src}/flashdreams", timeout=600)
ludus = f"{src}/flashdreams/integrations_v2/omnidreams/impl/ludus-renderer"
run("ludus deps", "pip install -q 'cmake>=4.0' 'ninja>=1.13.0' nvidia-nvcomp-cu12 loguru 2>&1 | tail -5", timeout=900)
run("ludus install", f"pip install -q --no-deps --no-build-isolation {ludus} 2>&1 | tail -20", timeout=900)
probe = (
    "import time, torch; t = time.time(); "
    "from ludus_renderer import LudusCudaTimestampedContext; "
    "ctx = LudusCudaTimestampedContext(device='cuda'); "
    "print('ludus context ready in %.1f s' % (time.time() - t))"
)
with open("/tmp/ludus_probe.py", "w") as handle:
    handle.write(probe)
run("libcuda", "mkdir -p /tmp/cudalink; ln -sf /usr/local/cuda-13.0/compat/libcuda.so.1 /tmp/cudalink/libcuda.so; ls -la /tmp/cudalink")
run("ludus cuda build", "cd /tmp && LIBRARY_PATH=/tmp/cudalink:$LIBRARY_PATH python3 /tmp/ludus_probe.py 2>&1 | grep -v -E -e '-MMD|-MD -MF'", timeout=2400)

say("\nrecon done")
log_file.close()
