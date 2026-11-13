from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.geometry_protocols import depth_configuration_identity


class WriteHeldoutDepthCalibrationRecordTest(unittest.TestCase):
    def test_writes_immutable_record_from_frozen_train32_candidates(self) -> None:
        configuration = {
            "model": "black-forest-labs/FLUX.2-klein-base-9B",
            "model_family": "klein",
            "transformer_path": "/tmp/checkpoint/transformer",
            "lora_path": None,
            "prompt_sha256": "prompt-digest",
            "negative_prompt_sha256": None,
            "steps": 28,
            "guidance_scale": 3.5,
            "seed": 0,
            "seed_mode": "fixed_per_sample_reset",
            "native_resolution": False,
            "depth_encoding": "relative_inverse_grayscale",
            "evaluator_sha256": "evaluator-digest",
            "runtime_backend_revision": "diffusers_environment",
            "attention_mode": None,
        }
        _, configuration_sha256 = depth_configuration_identity(configuration)
        subset = {
            "enabled": True,
            "protocol_id": "heldout_nyu_train32_v1",
            "manifest_path": "/tmp/heldout_nyu_train32_v1.jsonl",
            "manifest_sha256": "manifest-digest",
            "sample_count": 32,
            "full_resolved_sample_count": 249,
            "ordered_sample_ids_sha256": "sample-digest",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "nyu_train_results.json"
            per_sample = root / "nyu_train_per_sample.jsonl"
            output = root / "heldout_depth_affine_record.json"
            results.write_text(
                json.dumps(
                    {
                        "dataset": "nyu",
                        "requested_split": "train",
                        "resolved_splits": ["train"],
                        "images_evaluated": 32,
                        "sample_subset": subset,
                        "depth_evaluation": {
                            "calibration_candidates_only": True,
                            "configuration": configuration,
                            "configuration_sha256": configuration_sha256,
                        },
                    }
                ),
                encoding="utf-8",
            )
            per_sample.write_text(
                "".join(
                    json.dumps(
                        {
                            "dataset": "nyu",
                            "split": "train",
                            "sample_id": f"train__sample_{index:02d}",
                            "disparity_affine_abs_rel": 0.10,
                            "linear_affine_abs_rel": 0.20,
                        }
                    )
                    + "\n"
                    for index in range(32)
                ),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/evaluation/depth/write_heldout_depth_calibration_record.py"),
                    "--results",
                    str(results),
                    "--per-sample",
                    str(per_sample),
                    "--output",
                    str(output),
                ],
                check=True,
                cwd=ROOT,
            )
            record = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(record["selected_affine_space"], "disparity")
        self.assertEqual(record["configuration_sha256"], configuration_sha256)
        self.assertEqual(record["sample_count"], 32)
        self.assertEqual(record["calibration_subset"], subset)


if __name__ == "__main__":
    unittest.main()
