"""Frozen, model-facing prompts for neutral geometry evaluation protocols."""

from __future__ import annotations

import hashlib


CANONICAL_DEPTH_PROMPT = (
    "[TASK: DEPTH_ESTIMATION] Estimate dense metric projective camera-z depth in meters "
    "for the input image. Encode the result using the Vision Banana RGB depth encoding, "
    "with nearer valid depths at the near end of the encoding path and farther valid "
    "depths at the far end. Output only the RGB depth visualization with no text, labels, "
    "legends, or extra objects."
)

BARRON_SCALAR_DEPTH_PROMPT = (
    "[TASK: DEPTH_ESTIMATION] Estimate dense metric projective camera-z depth in meters "
    "for the input image. Encode depth as neutral grayscale (R=G=B) using the bounded "
    "scalar t=1-(1+d/10)^-2, where d is depth in meters, so nearer valid depths are "
    "darker and farther valid depths are brighter. Output only the registered grayscale "
    "depth map with no text, labels, legends, or extra objects."
)

CANONICAL_NORMAL_PROMPT = (
    "[TASK: SURFACE_NORMAL_ESTIMATION] Estimate a dense camera-space surface-normal map "
    "for the input image using this fixed convention: +x points image-right, +y points "
    "image-up, and +z points toward the camera/out of the image. Encode x, y, z in red, "
    "green, blue by mapping [-1, 1] to [0, 255]. Output only the clean RGB normal map "
    "with no text, labels, legends, or extra objects."
)

# The ``vision_banana_v3`` normal-map contract. CANONICAL_NORMAL_PROMPT above
# stays immutable for geometry-eval-v2 reproduction.
VISION_BANANA_NORMAL_PROMPT = (
    "[TASK: SURFACE_NORMAL_ESTIMATION] Estimate a dense camera-space surface-normal map "
    "for the input image using this fixed convention: +x points image-right, +y points "
    "image-up, and +z points toward the camera/out of the image. Encode the physical "
    "normal (x,y,z) as R=(1-x)/2, G=(1+y)/2, B=(1+z)/2, scaled to [0,255], so "
    "left-facing surfaces are pinkish red, upward-facing surfaces are light green, "
    "and camera-facing surfaces are light blue. Output only the clean RGB normal map "
    "with no text, labels, legends, or extra objects."
)

# Secondary prompt-only sensitivity checks. They retain the canonical physical
# representation while adding explicit spatial-fidelity requirements.
GEOMETRY_DETAIL_DEPTH_PROMPT = (
    "[TASK: DEPTH_ESTIMATION] Estimate dense metric projective camera-z depth in meters "
    "for the input image. Encode the result using the Vision Banana RGB depth encoding, "
    "with nearer valid depths at the near end of the encoding path and farther valid "
    "depths at the far end. Preserve fine structures, thin elements, and crisp object "
    "boundaries; do not blur across depth discontinuities. Output only the RGB depth "
    "visualization with no text, labels, legends, or extra objects."
)

GEOMETRY_DETAIL_NORMAL_PROMPT = (
    "[TASK: SURFACE_NORMAL_ESTIMATION] Estimate a dense camera-space surface-normal map "
    "for the input image using this fixed convention: +x points image-right, +y points "
    "image-up, and +z points toward the camera/out of the image. Encode x, y, z in red, "
    "green, blue by mapping [-1, 1] to [0, 255]. Preserve large-scale geometry, fine "
    "structural details, and crisp orientation boundaries. Output only the clean RGB "
    "normal map with no text, labels, legends, or extra objects."
)

VISION_BANANA_GEOMETRY_DETAIL_NORMAL_PROMPT = (
    "[TASK: SURFACE_NORMAL_ESTIMATION] Estimate a dense camera-space surface-normal map "
    "for the input image using this fixed convention: +x points image-right, +y points "
    "image-up, and +z points toward the camera/out of the image. Encode the physical "
    "normal (x,y,z) as R=(1-x)/2, G=(1+y)/2, B=(1+z)/2, scaled to [0,255], so "
    "left-facing surfaces are pinkish red, upward-facing surfaces are light green, "
    "and camera-facing surfaces are light blue. Preserve large-scale geometry, fine "
    "structural details, and crisp orientation boundaries. Output only the clean RGB "
    "normal map with no text, labels, legends, or extra objects."
)

# Concise zero-shot prompts use common geometry language but specify the
# representation and spatial contract precisely. They are intentionally short
# enough for prompt-sensitive image editors.
CONCISE_INVERSE_DEPTH_PROMPT = (
    "Create a dense relative inverse-depth map of the input as neutral grayscale "
    "(R=G=B): closer points are brighter and farther points are darker. Use one global "
    "mapping across the full image, preserve thin structures and sharp occlusion "
    "boundaries, keep smooth surfaces coherent, and return only the registered depth map."
)

CONCISE_LINEAR_DEPTH_PROMPT = (
    "Create a dense relative camera-depth map of the input as neutral grayscale "
    "(R=G=B): smaller depth is darker and larger depth is brighter. Use one global "
    "mapping across the full image, preserve thin structures and sharp occlusion "
    "boundaries, keep smooth surfaces coherent, and return only the registered depth map."
)

CONCISE_NORMAL_PROMPT = (
    "Create a dense camera-space unit surface-normal map registered to the input. Use "
    "+x image-right, +y image-up, +z toward the camera and encode (x,y,z) as "
    "R=(1-x)/2, G=(1+y)/2, B=(1+z)/2. Preserve planes, curved surfaces, thin "
    "structures, and sharp orientation boundaries; return only the RGB normal map."
)

# Ablation prompts are additive: older protocol variants keep their exact
# strings and hashes. The three inverse-depth prompts share one output
# representation; the paper-reconstructed RGB condition is a separate one.
DETAILED_INVERSE_DEPTH_PROMPT = (
    "Estimate a dense relative inverse-depth map for the input image. Produce exactly "
    "one output value for every input pixel at the same pixel location, with closer "
    "points brighter and farther points darker. Encode the result as neutral grayscale "
    "(R=G=B) using one consistent mapping over the complete image. Preserve object "
    "silhouettes, thin structures, occlusion boundaries, smooth surfaces, and small "
    "depth changes. Do not alter the scene or add texture, shading, color, text, labels, "
    "legends, borders, or objects; return only the registered depth map."
)

PAPER_RECONSTRUCTED_INVERSE_DEPTH_PROMPT = (
    "Generate the dense depth representation corresponding to the input image. Return "
    "a pixel-aligned neutral grayscale inverse-depth map (R=G=B), with nearer geometry "
    "brighter and farther geometry darker under one image-wide mapping. Preserve the "
    "input composition and fine boundaries, and output only the depth map without text, "
    "labels, legends, or additional objects."
)

PAPER_RECONSTRUCTED_BARRON_DEPTH_PROMPT = (
    "Generate a dense metric projective camera-depth visualization corresponding to "
    "the input image. Encode camera-z depth in meters with the declared bounded RGB "
    "depth transform, maintaining exact pixel-to-pixel registration and crisp depth "
    "boundaries. Return only the RGB depth visualization without text, labels, legends, "
    "or additional objects."
)

DETAILED_NORMAL_PROMPT = (
    "Estimate a dense camera-space unit surface normal for every input pixel at the "
    "same pixel location. Use +x image-right, +y image-up, and +z toward the camera. "
    "Encode physical (x,y,z) as R=(1-x)/2, G=(1+y)/2, B=(1+z)/2. Preserve planar and "
    "curved geometry, thin structures, and sharp orientation discontinuities. Do not "
    "copy image texture, change the scene, or add text, labels, legends, borders, or "
    "objects; return only the registered RGB normal map."
)

PAPER_RECONSTRUCTED_NORMAL_PROMPT = (
    "Generate the dense camera-space surface-normal representation corresponding to "
    "the input image. Use +x image-right, +y image-up, +z toward the camera and encode "
    "the unit vector as R=(1-x)/2, G=(1+y)/2, B=(1+z)/2. Maintain pixel alignment and "
    "surface boundaries, and return only the RGB normal map without text, labels, "
    "legends, or additional objects."
)


def prompt_sha256(prompt: str) -> str:
    """Return the UTF-8 SHA-256 used in immutable evaluation metadata."""

    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


CANONICAL_DEPTH_PROMPT_SHA256 = prompt_sha256(CANONICAL_DEPTH_PROMPT)
BARRON_SCALAR_DEPTH_PROMPT_SHA256 = prompt_sha256(BARRON_SCALAR_DEPTH_PROMPT)
CANONICAL_NORMAL_PROMPT_SHA256 = prompt_sha256(CANONICAL_NORMAL_PROMPT)
VISION_BANANA_NORMAL_PROMPT_SHA256 = prompt_sha256(VISION_BANANA_NORMAL_PROMPT)
GEOMETRY_DETAIL_DEPTH_PROMPT_SHA256 = prompt_sha256(
    GEOMETRY_DETAIL_DEPTH_PROMPT
)
GEOMETRY_DETAIL_NORMAL_PROMPT_SHA256 = prompt_sha256(
    GEOMETRY_DETAIL_NORMAL_PROMPT
)
VISION_BANANA_GEOMETRY_DETAIL_NORMAL_PROMPT_SHA256 = prompt_sha256(
    VISION_BANANA_GEOMETRY_DETAIL_NORMAL_PROMPT
)
CONCISE_INVERSE_DEPTH_PROMPT_SHA256 = prompt_sha256(CONCISE_INVERSE_DEPTH_PROMPT)
CONCISE_LINEAR_DEPTH_PROMPT_SHA256 = prompt_sha256(CONCISE_LINEAR_DEPTH_PROMPT)
CONCISE_NORMAL_PROMPT_SHA256 = prompt_sha256(CONCISE_NORMAL_PROMPT)
DETAILED_INVERSE_DEPTH_PROMPT_SHA256 = prompt_sha256(DETAILED_INVERSE_DEPTH_PROMPT)
PAPER_RECONSTRUCTED_INVERSE_DEPTH_PROMPT_SHA256 = prompt_sha256(
    PAPER_RECONSTRUCTED_INVERSE_DEPTH_PROMPT
)
PAPER_RECONSTRUCTED_BARRON_DEPTH_PROMPT_SHA256 = prompt_sha256(
    PAPER_RECONSTRUCTED_BARRON_DEPTH_PROMPT
)
DETAILED_NORMAL_PROMPT_SHA256 = prompt_sha256(DETAILED_NORMAL_PROMPT)
PAPER_RECONSTRUCTED_NORMAL_PROMPT_SHA256 = prompt_sha256(
    PAPER_RECONSTRUCTED_NORMAL_PROMPT
)
