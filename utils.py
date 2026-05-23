import os
import sys
import zipfile
import subprocess
import requests
import tempfile
import getpass
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

BASE_DIR = Path(__file__).resolve().parent

if os.name == "nt":
    WORK_DIR = BASE_DIR
else:
    WORK_DIR = Path(tempfile.gettempdir()) / f"ecg_{getpass.getuser()}"

BIN_DIR = WORK_DIR / "bin"
TEMP_DIR = WORK_DIR / "temp"

BIN_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def get_executable_name() -> str:
    if os.name == "nt":
        return "realesrgan-ncnn-vulkan.exe"
    return "realesrgan-ncnn-vulkan"


def get_executable_path():
    exec_name = get_executable_name()

    for path in BIN_DIR.rglob(exec_name):
        if path.is_file():
            return path

    return None


def download_realesrgan_binary(status_callback=None):
    exec_path = get_executable_path()

    if exec_path and exec_path.exists():
        if os.name != "nt":
            os.chmod(exec_path, 0o755)
        return exec_path

    if os.name == "nt":
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip"
    elif sys.platform == "darwin":
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-macos.zip"
    else:
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip"

    zip_path = BIN_DIR / "realesrgan-binary.zip"

    if status_callback:
        status_callback("Đang tải Real-ESRGAN binary...", 0.1)

    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0
    block_size = 1024 * 128

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)

                if total_size > 0 and status_callback:
                    progress = 0.1 + (downloaded / total_size) * 0.7
                    status_callback(
                        f"Đang tải: {downloaded / (1024 * 1024):.2f}MB / {total_size / (1024 * 1024):.2f}MB",
                        min(progress, 0.8),
                    )

    if status_callback:
        status_callback("Đang giải nén Real-ESRGAN...", 0.85)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(BIN_DIR)

    if zip_path.exists():
        zip_path.unlink()

    exec_path = get_executable_path()

    if not exec_path:
        raise FileNotFoundError("Không tìm thấy Real-ESRGAN executable sau khi giải nén.")

    if os.name != "nt":
        os.chmod(exec_path, 0o755)

    if status_callback:
        status_callback("Real-ESRGAN đã sẵn sàng.", 1.0)

    return exec_path


def pillow_enhance(input_path: str, output_path: str, scale: int = 4) -> str:
    from PIL import (
        Image,
        ImageEnhance,
        ImageFilter,
        ImageOps,
    )

    img = Image.open(input_path).convert("RGB")

    # cân bằng sáng nhẹ
    img = ImageOps.autocontrast(img, cutoff=1)

    # upscale trước
    new_size = (img.width * 4, img.height * 4)
    img = img.resize(new_size, Image.Resampling.LANCZOS)

    # sharpen vừa phải
    img = img.filter(
        ImageFilter.UnsharpMask(
            radius=1.2,
            percent=180,
            threshold=2,
        )
    )

    # clarity nhẹ
    img = ImageEnhance.Sharpness(img).enhance(2.2)

    # contrast nhẹ thôi
    img = ImageEnhance.Contrast(img).enhance(1.08)

    img.save(output_path, "PNG")

    return "Natural ECG enhancement"


def run_realesrgan(
    input_path: str,
    output_path: str,
    model_name: str = "realesrgan-x4plus",
    tile_size: int = 100,
    scale: int = 4,
) -> str:
    scale = 4

    exec_path = get_executable_path()

    if not exec_path or not exec_path.exists():
        exec_path = download_realesrgan_binary()

    exec_dir = exec_path.parent

    cmd = [
        str(exec_path),
        "-i", str(input_path),
        "-o", str(output_path),
        "-n", str(model_name),
        "-s", "4",
        "-t", str(tile_size),
        "-g", "-1",
    ]

    process = subprocess.run(
        cmd,
        cwd=str(exec_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )

    if process.returncode == 0 and Path(output_path).exists():
        return f"Real-ESRGAN {model_name} x4"

    error_msg = process.stderr or process.stdout or ""

    if "invalid gpu device" in error_msg.lower() or process.returncode == 255:
        pillow_enhance(input_path, output_path, scale=4)
        return "Advanced ECG Pillow fallback x4 vì Cloud không hỗ trợ Vulkan GPU"

    raise RuntimeError(f"Real-ESRGAN failed code {process.returncode}: {error_msg}")