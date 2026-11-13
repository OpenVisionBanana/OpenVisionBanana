import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from common.surface_normal_encoding import (
    VISION_BANANA_NORMAL_ENCODING,
    decode_surface_normals_rgb,
    encode_surface_normals_rgb,
)

try:
    import train_dreambooth_partial_flux2_klein_img2img as klein_trainer
    from train_dreambooth_partial_flux2_klein_img2img import (
        BucketBatchSampler,
        DatasetSpec,
        DreamBoothDataset,
        PreparedRow,
        WeightedDatasetPool,
    )
except ImportError:  # pragma: no cover
    klein_trainer = None
    BucketBatchSampler = None
    DatasetSpec = None
    DreamBoothDataset = None
    PreparedRow = None
    WeightedDatasetPool = None


@unittest.skipIf(WeightedDatasetPool is None, "Klein trainer dependencies are not installed")
class KleinPoolIntegrationTest(unittest.TestCase):
    def test_final_normal_validity_rejects_neutral_fill_and_nonfinite(self):
        torch=klein_trainer.torch
        target=torch.zeros((3,2,2))
        target[:,0,0]=torch.tensor([0.,0.,1.])
        target[:,0,1]=1/255  # Neutral 128 RGB in the VAE range.
        target[:,1,0]=torch.tensor([float('nan'),0.,1.])
        target[:,1,1]=torch.tensor([1.,0.,0.])
        mask=klein_trainer.final_normal_target_validity(target)
        assert mask.tolist()==[[[True,False],[False,True]]]

    def test_neutral_source_normals_are_not_normalized_into_a_unit_direction(self):
        resized=klein_trainer.resize_normal_vector_then_encode(
            Image.new('RGB',(4,4),(128,128,128)),(8,8),VISION_BANANA_NORMAL_ENCODING,
            Image.new('L',(4,4),255))
        _,valid=decode_surface_normals_rgb(np.asarray(resized),encoding=VISION_BANANA_NORMAL_ENCODING)
        assert not valid.any()

    def test_masked_loader_intersects_final_normal_validity(self):
        torch=klein_trainer.torch
        data=DreamBoothDataset.__new__(DreamBoothDataset)
        data.instance_prompt='fallback'
        data.args=SimpleNamespace(image_column='image',cond_image_column='conditioning_image',loss_mask_policy='auto',
            depth_target_resize_policy='metric_field_then_encode',normal_target_resize_policy='vector_field_renormalize_then_encode',
            normal_target_encoding='vision_banana_x_left_red',depth_target_encoding='relative_inverse_grayscale',
            depth_prompt_contract='campaign_detailed_inverse',center_crop=False,random_flip=False)
        image=Image.new('RGB',(2,2),(128,128,255))
        data._load_row_image=Mock(return_value=image)
        data._load_row_loss_mask=Mock(return_value=Image.new('L',(2,2),255))
        target=torch.zeros((3,2,2));target[2]=1.;target[:,0,0]=1/255
        data.paired_transform=Mock(return_value=(target,target.clone(),torch.ones((1,2,2))))
        row=PreparedRow('normals',None,'example','input.png','normal.png','normal prompt',loss_mask='mask.png',
            task='surface_normal_estimation',normal_encoding='vision_banana_x_left_red')
        example=data._load_example(row,0,0,(2,2))
        assert example['loss_mask'].tolist()==[[[0.,1.],[1.,1.]]]
        torch.testing.assert_close(example['instance_images'],target)

    def test_effective_prompt_matches_representation_specific_depth_cache_key(self):
        depth = SimpleNamespace(text="row metric-depth prompt", task="depth_estimation")
        expected = klein_trainer.CONCISE_INVERSE_DEPTH_PROMPT
        self.assertEqual(
            klein_trainer.effective_instance_prompt(
                depth, "fallback", "relative_inverse_grayscale", "canonical_v2"
            ),
            expected,
        )
        self.assertEqual(
            klein_trainer.effective_instance_prompt(
                depth, "fallback", "vision_banana_rgb", "canonical_v2"
            ),
            depth.text,
        )
        normal = SimpleNamespace(text="normal prompt", task="surface_normal_estimation")
        self.assertEqual(
            klein_trainer.effective_instance_prompt(
                normal, "fallback", "relative_inverse_grayscale", "canonical_v2"
            ),
            normal.text,
        )

    def test_decoded_angular_auxiliary_disables_vae_cpu_offload(self):
        base = dict(
            palette_manifold_loss_weight=0.0,
            metric_depth_aux_weight=0.0,
            unit_normal_loss_weight=0.0,
            angular_normal_aux_weight=0.0,
        )
        self.assertFalse(
            klein_trainer.decoded_geometry_auxiliary_enabled(SimpleNamespace(**base))
        )
        base["angular_normal_aux_weight"] = 0.05
        self.assertTrue(
            klein_trainer.decoded_geometry_auxiliary_enabled(SimpleNamespace(**base))
        )

    def test_camera_facing_v4_normal_rows_are_versioned_canonical(self):
        contract = klein_trainer.prepared_row_geometry_contract(
            {
                "task": "surface_normal_estimation",
                "source_dataset": "scannetv2",
                "metadata": {
                    "geometry_version": "geometry_camera_facing_v4",
                    "normal_orientation": "face_camera_ray",
                    "canonical_normal_convention": (
                        "camera_x_right_y_up_z_toward_camera"
                    ),
                    "normal_transform_already_applied": True,
                },
            }
        )
        self.assertEqual(contract["normal_contract_status"], "versioned_canonical")
        self.assertEqual(contract["stored_normal_permutation"], (0, 1, 2))
        self.assertEqual(contract["stored_normal_signs"], (1, 1, 1))

    def test_canonical_normal_horizontal_flip_reverses_space_and_x_axis(self):
        normals = np.asarray(
            [[[-0.8, 0.2, 0.56], [-0.3, -0.4, 0.86], [0.7, 0.1, 0.70]]],
            dtype=np.float32,
        )
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        encoded = encode_surface_normals_rgb(
            normals,
            np.ones(normals.shape[:2], dtype=bool),
            encoding=VISION_BANANA_NORMAL_ENCODING,
        )
        flipped = klein_trainer.horizontal_flip_canonical_normal_rgb(
            Image.fromarray(encoded, mode="RGB")
        )
        decoded, valid = decode_surface_normals_rgb(
            np.asarray(flipped),
            encoding=VISION_BANANA_NORMAL_ENCODING,
            min_norm=1e-6,
        )
        expected = normals[:, ::-1].copy()
        expected[..., 0] *= -1
        self.assertTrue(valid.all())
        np.testing.assert_allclose(decoded, expected, atol=0.012)

    def test_normal_encoding_transcode_preserves_physical_vectors(self):
        normals = np.asarray(
            [[[0.6, -0.2, 0.77], [-0.5, 0.5, 0.707]]], dtype=np.float32
        )
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        legacy = encode_surface_normals_rgb(
            normals,
            np.ones(normals.shape[:2], dtype=bool),
            encoding="legacy_direct_xyz",
        )
        converted = klein_trainer.convert_surface_normal_image_encoding(
            Image.fromarray(legacy, mode="RGB"),
            "legacy_direct_xyz",
            VISION_BANANA_NORMAL_ENCODING,
        )
        decoded, valid = decode_surface_normals_rgb(
            np.asarray(converted),
            encoding=VISION_BANANA_NORMAL_ENCODING,
            min_norm=1e-6,
        )
        self.assertTrue(valid.all())
        np.testing.assert_allclose(decoded, normals, atol=0.012)

    def test_masked_normal_resize_does_not_leak_invalid_neutral_vectors(self):
        normal = np.asarray([0.4, -0.2, 0.894], dtype=np.float32)
        normal /= np.linalg.norm(normal)
        vectors = np.broadcast_to(normal, (4, 4, 3)).copy()
        valid = np.zeros((4, 4), dtype=bool)
        valid[1:3, 1:3] = True
        encoded = encode_surface_normals_rgb(
            vectors, valid, encoding=VISION_BANANA_NORMAL_ENCODING
        )
        resized = klein_trainer.resize_normal_vector_then_encode(
            Image.fromarray(encoded, mode="RGB"),
            (12, 12),
            VISION_BANANA_NORMAL_ENCODING,
            Image.fromarray(valid.astype(np.uint8) * 255, mode="L"),
        )
        decoded, resized_valid = decode_surface_normals_rgb(
            np.asarray(resized),
            encoding=VISION_BANANA_NORMAL_ENCODING,
            min_norm=0.05,
        )
        self.assertGreater(int(resized_valid.sum()), 0)
        np.testing.assert_allclose(
            decoded[resized_valid],
            np.broadcast_to(normal, decoded[resized_valid].shape),
            atol=0.012,
        )

    def test_disabled_loss_masks_still_protect_normal_resize(self):
        dataset = DreamBoothDataset.__new__(DreamBoothDataset)
        dataset.args = SimpleNamespace(
            image_column="image",
            cond_image_column="conditioning_image",
            loss_mask_policy="disabled",
            depth_target_resize_policy="metric_field_then_encode",
            normal_target_resize_policy="vector_field_renormalize_then_encode",
            normal_target_encoding="vision_banana_x_left_red",
            depth_target_encoding="relative_inverse_grayscale",
            depth_prompt_contract="campaign_detailed_inverse",
            center_crop=False,
            random_flip=False,
        )
        dataset.instance_prompt = "fallback"
        target_image = object()
        conditioning_image = object()
        validity_mask = object()
        resized_target = object()
        dataset._load_row_image = Mock(side_effect=[target_image, conditioning_image])
        dataset._load_row_loss_mask = Mock(return_value=validity_mask)
        dataset.paired_transform = Mock(
            return_value=(resized_target, conditioning_image, None)
        )
        row = PreparedRow(
            dataset_name="normals",
            dataset_root=None,
            row_id="example",
            conditioning_image="conditioning.png",
            image="target.png",
            text="normal prompt",
            loss_mask="validity.png",
            task="surface_normal_estimation",
            source_dataset="nyuv2",
            preprocess_target="resize_normal_vector_then_encode",
            normal_encoding="vision_banana_x_left_red",
        )

        with patch.object(
            klein_trainer,
            "convert_surface_normal_image_encoding",
            return_value=target_image,
        ), patch.object(
            klein_trainer,
            "resize_normal_vector_then_encode",
            return_value=resized_target,
        ) as resize_normal:
            example = dataset._load_example(row, 0, 0, (512, 512))

        dataset._load_row_loss_mask.assert_called_once_with(row)
        resize_normal.assert_called_once_with(
            target_image,
            (512, 512),
            "vision_banana_x_left_red",
            validity_mask,
        )
        self.assertNotIn("loss_mask", example)
        self.assertIsNone(dataset.paired_transform.call_args.kwargs["loss_mask"])
        self.assertFalse(
            dataset.paired_transform.call_args.kwargs["fill_missing_loss_mask"]
        )

    def test_latent_loss_mask_area_downsampling_hard_and_soft(self):
        torch = klein_trainer.torch
        mask = torch.tensor(
            [[[[1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]]]],
            dtype=torch.float32,
        )
        soft = klein_trainer.prepare_latent_loss_mask(
            mask, (2, 2), mode="soft", min_valid_fraction=0.5
        )
        expected_soft = torch.tensor([[[[1.0, 0.0], [0.25, 0.25]]]])
        torch.testing.assert_close(soft, expected_soft)
        hard = klein_trainer.prepare_latent_loss_mask(
            mask, (2, 2), mode="hard", min_valid_fraction=0.5
        )
        torch.testing.assert_close(
            hard, torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
        )

    def test_latent_loss_mask_erosion_and_empty_sample_rejection(self):
        torch = klein_trainer.torch
        mask = torch.zeros((2, 1, 5, 5), dtype=torch.float32)
        mask[1, :, 1:4, 1:4] = 1.0
        latent = klein_trainer.prepare_latent_loss_mask(
            mask, (5, 5), mode="hard", erosion_radius=1
        )
        self.assertEqual(float(latent[0].sum()), 0.0)
        self.assertEqual(float(latent[1].sum()), 1.0)
        expanded = latent.expand(2, 4, 5, 5)
        counts, fractions, eligible = klein_trainer.eligible_loss_mask_samples(
            mask, expanded, min_sample_valid_fraction=0.1
        )
        torch.testing.assert_close(counts, torch.tensor([0.0, 4.0]))
        torch.testing.assert_close(fractions, torch.tensor([0.0, 9.0 / 25.0]))
        self.assertEqual(eligible.tolist(), [False, True])

    def test_cap_then_replacement_sampling_uses_stable_full_pool_indices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for image_name in ("source-a.png", "source-b.png"):
                for expression_index in range(8):
                    rows.append(
                        {
                            "id": f"{image_name}-{expression_index}",
                            "image": f"target-{image_name}-{expression_index}.png",
                            "conditioning_image": image_name,
                            "text": f"expression {expression_index}",
                            "task": "refseg",
                            "metadata": {"referential_expression": f"expression {expression_index}"},
                        }
                    )
            (root / "metadata.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            pool = WeightedDatasetPool(
                [DatasetSpec("refseg", root, 1.0)],
                samples_per_epoch=100,
                seed=42,
                image_column="image",
                cond_image_column="conditioning_image",
                caption_column="text",
                sampling_strategy="with_replacement",
                max_refseg_expressions_per_image=5,
            )
            self.assertEqual(len(pool.rows), 10)
            first = pool.sample_epoch_indices(0)
            self.assertEqual(first, pool.sample_epoch_indices(0))
            self.assertEqual(len(first), 100)
            self.assertLessEqual(len(set(first)), 10)
            self.assertNotEqual(first, pool.sample_epoch_indices(1))

    def test_bucket_padding_keeps_epoch_batch_count_fixed(self):
        class Pool:
            samples_per_epoch = 10

            def sample_epoch_indices(self, epoch):
                return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        class Dataset:
            dataset_pool = Pool()
            resample_dataset_each_epoch = True
            rows = [
                PreparedRow("a" if index < 5 else "b", None, str(index), None, "x", "prompt")
                for index in range(10)
            ]
            pixel_values = [(None, 0 if index < 5 else 1) for index in range(10)]

            def set_epoch(self, epoch):
                self.epoch = epoch

            def _log_sample_counts(self, epoch):
                pass

        sampler = BucketBatchSampler(
            Dataset(), batch_size=4, drop_last=True, task_homogeneous_batches=True, seed=42
        )
        self.assertEqual(len(sampler), 3)
        self.assertTrue(all(len(batch) == 4 for batch in sampler))
        sampler.set_epoch(1)
        self.assertEqual(len(sampler), 3)

    def test_fixed_pool_sampler_uses_only_materialized_trial_rows(self):
        class Pool:
            samples_per_epoch = 6

            def sample_epoch_indices(self, epoch):
                raise AssertionError("fixed trials must not sample from the full pool again")

        class Dataset:
            dataset_pool = Pool()
            resample_dataset_each_epoch = False
            rows = [
                PreparedRow("trial", None, str(index), None, "x", "prompt")
                for index in range(6)
            ]
            pixel_values = [(None, 0) for _ in rows]

            def set_epoch(self, epoch):
                self.epoch = epoch

            def _log_sample_counts(self, epoch):
                pass

        sampler = BucketBatchSampler(
            Dataset(), batch_size=2, drop_last=True, task_homogeneous_batches=True, seed=42
        )
        self.assertEqual(len(sampler), 3)
        self.assertEqual(sorted(index for batch in sampler for index in batch), list(range(6)))

if __name__ == "__main__":
    unittest.main()
