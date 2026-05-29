import cv2
from pathlib import Path

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


UPSAMPLER = None


def get_upsampler():
    global UPSAMPLER

    if UPSAMPLER is not None:
        return UPSAMPLER

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )

    model_path = Path("weights/RealESRGAN_x4plus.pth")

    UPSAMPLER = RealESRGANer(
        scale=4,
        model_path=str(model_path),
        model=model,
        tile=256,
        tile_pad=10,
        pre_pad=0,
        half=False,
    )

    return UPSAMPLER


def enhance_with_realesrgan(image_rgb):
    upsampler = get_upsampler()

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    output_bgr, _ = upsampler.enhance(
        image_bgr,
        outscale=4
    )

    output_rgb = cv2.cvtColor(
        output_bgr,
        cv2.COLOR_BGR2RGB
    )

    return output_rgb