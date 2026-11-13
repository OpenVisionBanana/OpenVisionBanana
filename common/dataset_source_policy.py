"""Reject temporary project dataset uploads, including public test repositories.

Author repositories and third-party public distributions remain usable. This
policy does not grant licenses, accept access terms or bypass gated access.
It applies to dataset acquisition, not the project's model distribution.
"""
from urllib.parse import unquote, urlsplit


def validate_dataset_source(value):
    value = str(value)
    if "://" in value:
        parsed = urlsplit(value)
        if (parsed.hostname or "").lower() not in {"huggingface.co", "www.huggingface.co", "hf.co"}:
            return value
        parts = unquote(parsed.path).strip("/").split("/")
        if parts[:2] == ["api", "datasets"]:
            parts = parts[2:]
        elif parts and parts[0] in {"datasets", "spaces"}:
            parts = parts[1:]
    else:
        parts = value.strip("/").split("/")
    owner = parts[0].casefold() if parts else ""
    repository = parts[1].casefold() if len(parts) > 1 else ""
    if (owner in {"ovb26", "funailab"} or "arpit" in owner
            or "visionbanana" in owner.replace("-", "_").replace("_", "")
            or owner.startswith("ovb") or repository.startswith("ovb_")
            or repository in {"ovb-data", "ovbdata"}):
        raise ValueError("Temporary project/test dataset repositories are prohibited. "
                         "Use original or independently published public dataset sources.")
    return value
