import contextlib
import subprocess
import time
from pathlib import Path

from tritonclient.http import InferenceServerClient
from ultralytics.models.yolo import YOLO

runtime = "docker"

# 1) Export YOLO to TensorRT Format

model = YOLO("yolo26n.pt")

engine_file = model.export(format="engine", dynamic=True)


# 2) Setting Up Triton Model Repository

model_name = "yolo"
triton_repo_path = Path("tmp") / "triton_repo"
triton_model_path = triton_repo_path / model_name

(triton_model_path / "1").mkdir(parents=True, exist_ok=True)

Path(engine_file).rename(triton_model_path / "1" / "model.plan")

(triton_model_path / "config.pbtxt").touch()

data = f"""name: "{model_name}"
platform: "tensorrt_plan"
max_batch_size: 16
"""

with open(triton_model_path / "config.pbtxt", "w") as f:
    f.write(data)


# 3) Running Triton Inference Server

tag = "nvcr.io/nvidia/tritonserver:26.02-py3"

subprocess.call(f"{runtime} pull {tag}", shell=True)

gpu_flags = (
    "--device nvidia.com/gpu=all"
    if runtime == "podman"
    else "--runtime=nvidia --gpus all"
)

container_name = "triton_server"

subprocess.call(
    f"{runtime} run -d --rm --name {container_name} {gpu_flags} -v {triton_repo_path.absolute()}:/models:z -p 8299:8000 {tag} tritonserver --model-repository=/models",
    shell=True,
)

triton_client = InferenceServerClient(url="127.0.0.1:8299", verbose=False, ssl=False)

for _ in range(30):
    with contextlib.suppress(Exception):
        if triton_client.is_model_ready(model_name):
            break
    time.sleep(2)
else:
    raise RuntimeError("Model failed to load")
