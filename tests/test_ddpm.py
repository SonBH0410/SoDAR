"""Small CPU-friendly joint generator integration test."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch not installed")
class JointDiffusionTests(unittest.TestCase):
    def test_four_stage_unet_loss_backward_and_ddim(self):
        import torch
        from sodar.diffusion.model import JointUNet
        from sodar.diffusion.process import JointGaussianDiffusion
        torch.set_num_threads(min(4, torch.get_num_threads()))
        model = JointUNet(dim=16, dim_mults=(1, 2, 4, 8), heads=2, dim_head=8)
        diffusion = JointGaussianDiffusion(model, image_size=64, timesteps=16,
                                           sampling_timesteps=4)
        losses = diffusion.training_losses(torch.rand(1, 3, 64, 64),
                                           (torch.rand(1, 1, 64, 64)>0.5).float())
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        self.assertIsNotNone(model.final_conv.weight.grad)
        diffusion.eval()
        imgs, logits = diffusion.sample(1)
        self.assertEqual(imgs.shape, (1, 3, 64, 64))
        self.assertEqual(logits.shape, (1, 2, 64, 64))

    def test_checkpoint_sampling_resume_and_contract(self):
        from sodar.phantom import make_phantom
        from sodar.diffusion.data import ddpm_config
        from sodar.diffusion.runner import train_generator, load_checkpoint, sample_generator
        from scripts.audit_generated import run_audit

        with tempfile.TemporaryDirectory(prefix="sodar-ddpm-test-") as tmp:
            root = Path(tmp)
            make_phantom(root)
            cfg = ddpm_config(path=str(Path(__file__).resolve().parents[1]/"configs"/"ddpm_phantom.yaml"),
                              data_root=str(root/"real"), manifest=str(root/"manifest.json"),
                              save_root=str(root/"checkpoints"), steps=2)
            checkpoint = train_generator(cfg)
            self.assertEqual(load_checkpoint(checkpoint)["step"], 2)
            new_config = ddpm_config(path=str(Path(__file__).resolve().parents[1]/"configs"/"ddpm_phantom.yaml"),
                                     data_root=str(root/"real"), manifest=str(root/"manifest.json"),
                                     save_root=str(root/"checkpoints"), steps=3)
            checkpoint = train_generator(new_config, resume=str(checkpoint))
            self.assertEqual(load_checkpoint(checkpoint)["step"], 3)
            output = root / "diffusion_generated"
            result = sample_generator(checkpoint, output=output, num_samples=3,
                                      batch_size=2, seed=999)
            self.assertEqual(result["count"], 3)
            self.assertEqual(run_audit(output, expected_count=3)["result"], "PASS")
            from sodar.config import configure
            from sodar.io import audit
            student_conf = configure("S4", config_path=str(Path(__file__).resolve().parents[1]/"configs"/"phantom.yaml"),
                                     data_root=str(root/"real"), generated_root=str(output))
            student_conf["data"]["manifest"] = str(root/"manifest.json")
            student_conf["data"]["require_synthetic_provenance"] = True
            self.assertTrue(audit(student_conf, check_all_arrays=False)["synthetic"]["train_manifest_verified"])
            student_conf["data"]["synthetic_format"] = "probabilities"
            with self.assertRaisesRegex(ValueError, "double-softmaxed"):
                audit(student_conf, check_all_arrays=False)
            with self.assertRaises(FileExistsError):
                sample_generator(checkpoint, output=output, num_samples=3,
                                 batch_size=2, seed=999)
            resumed = sample_generator(checkpoint, output=output, num_samples=3,
                                       batch_size=2, seed=999, resume_output=True)
            self.assertEqual(resumed["count"], 3)


if __name__ == "__main__":
    unittest.main()
