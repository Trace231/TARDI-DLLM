from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageFilter, ImageEnhance


ROOT = Path(__file__).resolve().parents[1]
ATTACHMENT_PATH = Path("/Users/thomaswang/Downloads/附件4：中国版图.JPG")
OUTPUT_DIR = ROOT / "output" / "china_population_art"

# Province extraction is now attachment-driven instead of GeoJSON-driven.
# These values are hand-tuned from the actual attachment so the resulting masks
# follow the original pixel boundaries instead of an approximate projection.
PROVINCE_SPECS = {
    "甘肃": {
        "seed": (3013, 2207),
        "bbox": (1858, 1914, 3541, 3347),
        "threshold": 42,
        "overlay": "#7DA7FF",
    },
    "辽宁": {
        "seed": (4850, 2428),
        "bbox": (4582, 1816, 5296, 2486),
        "threshold": 34,
        "overlay": "#5A9AF0",
    },
    "河北": {
        "seed": (4241, 2441),
        "bbox": (4029, 1939, 4688, 2861),
        "threshold": 34,
        "overlay": "#7CAEFF",
    },
    "山东": {
        "seed": (4608, 2651),
        "bbox": (4168, 2535, 4980, 3096),
        "threshold": 34,
        "overlay": "#2D79E5",
    },
    "天津": {
        "seed": (4512, 2565),
        "bbox": (4300, 2290, 4540, 2620),
        "threshold": 42,
        "overlay": "#8C68FF",
    },
    "江西": {
        "seed": (4205, 4022),
        "bbox": (4041, 3701, 4545, 4486),
        "threshold": 34,
        "overlay": "#7AA7FF",
    },
    "广东": {
        "seed": (4072, 4782),
        "bbox": (3638, 4342, 4412, 5086),
        "threshold": 40,
        "overlay": "#1764CC",
    },
}


def extract_mask(
    work: np.ndarray,
    size: tuple[int, int],
    seed: tuple[int, int],
    bbox: tuple[int, int, int, int],
    threshold: int,
) -> Image.Image:
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - 10)
    y1 = max(0, y1 - 10)
    x2 = min(size[0] - 1, x2 + 10)
    y2 = min(size[1] - 1, y2 + 10)

    sx, sy = seed
    seed = work[sy, sx]

    queue = deque([(sx, sy)])
    seen = np.zeros((y2 - y1 + 1, x2 - x1 + 1), dtype=bool)
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)

    while queue:
        x, y = queue.popleft()
        if not (x1 <= x <= x2 and y1 <= y <= y2):
            continue

        lx, ly = x - x1, y - y1
        if seen[ly, lx]:
            continue
        seen[ly, lx] = True

        rgb = work[y, x]
        if rgb.mean() > 248:
            continue
        if np.abs(rgb - seed).sum() > threshold:
            continue

        mask[y, x] = 255
        queue.extend(((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)))

    mask_img = Image.fromarray(mask, mode="L")
    # Fill label holes and smooth jaggedness without moving the outer boundary much.
    mask_img = mask_img.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
    mask_img = mask_img.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    return mask_img


def resize_specs(specs: dict[str, dict], scale_x: float, scale_y: float) -> dict[str, dict]:
    resized: dict[str, dict] = {}
    for province, spec in specs.items():
        sx, sy = spec["seed"]
        x1, y1, x2, y2 = spec["bbox"]
        resized[province] = {
            **spec,
            "seed": (int(round(sx * scale_x)), int(round(sy * scale_y))),
            "bbox": (
                int(round(x1 * scale_x)),
                int(round(y1 * scale_y)),
                int(round(x2 * scale_x)),
                int(round(y2 * scale_y)),
            ),
        }
    return resized


def boundary(mask: Image.Image, width: int) -> Image.Image:
    expanded = mask.filter(ImageFilter.MaxFilter(width))
    shrunk = mask.filter(ImageFilter.MinFilter(max(3, width - 2)))
    return ImageChops.subtract(expanded, shrunk)


def apply_overlay(base: Image.Image, mask: Image.Image, color: str, fill_alpha: int, edge_alpha: int) -> Image.Image:
    out = base.copy()
    rgb = ImageColor.getrgb(color)

    fill = Image.new("RGBA", out.size, rgb + (0,))
    fill.putalpha(mask.point(lambda x: int(x * fill_alpha / 255)))
    out.alpha_composite(fill)

    glow = boundary(mask, 11).filter(ImageFilter.GaussianBlur(5))
    glow_layer = Image.new("RGBA", out.size, rgb + (0,))
    glow_layer.putalpha(glow.point(lambda x: min(220, int(x * edge_alpha / 255))))
    out.alpha_composite(glow_layer)

    crisp = boundary(mask, 5)
    crisp_layer = Image.new("RGBA", out.size, (255, 255, 255, 0))
    crisp_layer.putalpha(crisp.point(lambda x: min(180, int(x * 145 / 255))))
    out.alpha_composite(crisp_layer)
    return out


def build_debug_view(base: Image.Image, masks: dict[str, Image.Image]) -> Image.Image:
    debug = base.convert("RGBA")
    dim = Image.new("RGBA", debug.size, (255, 255, 255, 86))
    debug.alpha_composite(dim)
    for province, spec in PROVINCE_SPECS.items():
        debug = apply_overlay(debug, masks[province], spec["overlay"], fill_alpha=168, edge_alpha=150)
    return debug


def build_final_view(base: Image.Image, masks: dict[str, Image.Image]) -> Image.Image:
    result = base.convert("RGBA")
    for province, spec in PROVINCE_SPECS.items():
        fill_alpha = 118 if province != "天津" else 170
        edge_alpha = 125 if province != "天津" else 180
        result = apply_overlay(result, masks[province], spec["overlay"], fill_alpha=fill_alpha, edge_alpha=edge_alpha)
    result = ImageEnhance.Contrast(result).enhance(1.02)
    return ImageEnhance.Sharpness(result).enhance(1.04)


def main() -> None:
    if not ATTACHMENT_PATH.exists():
        raise FileNotFoundError(f"Missing source image: {ATTACHMENT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    attachment = Image.open(ATTACHMENT_PATH).convert("RGB")
    work_image = attachment.copy()
    work_image.thumbnail((1200, 1200))
    scale_x = work_image.size[0] / attachment.size[0]
    scale_y = work_image.size[1] / attachment.size[1]
    scaled_specs = resize_specs(PROVINCE_SPECS, scale_x, scale_y)
    smoothed = np.array(work_image.filter(ImageFilter.GaussianBlur(1.6))).astype(np.int16)

    masks: dict[str, Image.Image] = {}
    for province, spec in scaled_specs.items():
        mask_small = extract_mask(
            smoothed,
            work_image.size,
            spec["seed"],
            spec["bbox"],
            spec["threshold"],
        )
        masks[province] = mask_small.resize(attachment.size, Image.Resampling.BILINEAR).point(
            lambda x: 255 if x >= 96 else 0
        )

    final_map = build_final_view(attachment, masks)
    debug_map = build_debug_view(attachment, masks)

    final_path = OUTPUT_DIR / "china_survey_locations_attachment_exactmask.png"
    debug_path = OUTPUT_DIR / "china_survey_locations_attachment_debug.png"
    final_map.save(final_path)
    debug_map.save(debug_path)

    print(f"Final map: {final_path}")
    print(f"Debug map: {debug_path}")


if __name__ == "__main__":
    main()
