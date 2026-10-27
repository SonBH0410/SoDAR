"""Optional torch-backed unit tests. Skipped on thin CPUs without torch."""

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch not installed")
class TorchTests(unittest.TestCase):
    def test_unet_losses_and_meta_update(self):
        import torch
        from sodar.models import Unet
        from sodar.losses import hard_loss, synthetic_loss
        from sodar.weighting import LambdaWeight, meta_weight_step
        from sodar.config import configure

        conf = configure("S4", config_path="configs/phantom.yaml")
        model = Unet(base_channels=8)
        model.train()
        image = torch.randn(2, 3, 64, 64)
        mask = (torch.rand(2, 1, 64, 64) > .6).float()
        probs = torch.softmax(torch.randn(2, 2, 64, 64), dim=1)
        real = hard_loss(model(image), mask, .5)
        syn = synthetic_loss(model(image), {"prob": probs}, conf)
        self.assertTrue(torch.isfinite(real).item())
        self.assertTrue(torch.isfinite(syn).item())
        weight = LambdaWeight(.1, .5)
        opt = torch.optim.Adam([weight.alpha], lr=.01)
        first = float(weight().detach())
        meta_weight_step(model, weight, opt, real, syn,
                         {"image": image, "mask": mask}, conf, inner_lr=1e-3)
        second = float(weight().detach())
        self.assertTrue(0 < second < .5)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
