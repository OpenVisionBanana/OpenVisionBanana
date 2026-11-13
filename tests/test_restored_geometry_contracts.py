import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.mark.parametrize("source,version", [
    ("nyuv2", "nyuv2_geonet_dsine_v1"), ("diode", "diode_identity_reproduced_v2")])
def test_restored_labels_are_canonical_without_runtime_axis_changes(source, version):
    from train_dreambooth_partial_flux2_klein_img2img import prepared_row_geometry_contract
    row = dict(task="surface_normals", source_dataset=source,
               metadata=dict(geometry_version=version,
                   canonical_normal_convention="camera_x_right_y_up_z_toward_camera"))
    contract = prepared_row_geometry_contract(row)
    assert contract["normal_contract_status"] == "versioned_canonical"
    assert contract["stored_normal_signs"] == (1,1,1)
    row["metadata"]["canonical_normal_convention"] = "unknown"
    assert prepared_row_geometry_contract(row)["normal_contract_status"] == "unverified"
    row["metadata"]["geometry_version"] = "diode_identity_candidate_unverified_v1"
    assert prepared_row_geometry_contract(row)["normal_contract_status"] == "unverified"


def test_restored_diode_source_guard_has_reproducible_evidence():
    from scripts.dataset_prep.finalize_training_recipe import ensure_diode_convention_verified
    ensure_diode_convention_verified()
