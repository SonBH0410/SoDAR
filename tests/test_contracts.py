"""Fast standard-library tests; do not require torch."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from sodar.config import configure
from sodar.io import audit, generated_pairs, load_manifest, probs_from_generated, read_two_channel_array
from sodar.phantom import make_phantom
from scripts.smoke_test import run_numpy_flow


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sodar-test-")
        self.root = Path(self.temp.name)
        make_phantom(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def conf(self):
        cfg = configure("S4", config_path=str(Path(__file__).resolve().parents[1]/"configs"/"phantom.yaml"),
                        data_root=str(self.root/"real"), generated_root=str(self.root/"generated"))
        cfg["data"]["manifest"] = str(self.root/"manifest.json")
        return cfg

    def test_pairs_and_manifest(self):
        report = audit(self.conf())
        self.assertEqual(report["patient_disjoint"], "yes")
        self.assertEqual(len(generated_pairs(self.conf())), 12)

    def test_temperature_and_hard_label(self):
        path = generated_pairs(self.conf())[0][1]
        logits = read_two_channel_array(path, expected_format="logits")
        p1 = probs_from_generated(logits, temperature=1, fmt="logits")
        p4 = probs_from_generated(logits, temperature=4, fmt="logits")
        self.assertTrue(np.allclose(p4.sum(0), 1))
        self.assertTrue(np.array_equal(p1.argmax(0), p4.argmax(0)))
        self.assertLess(np.abs(p4[1]-.5).mean(), np.abs(p1[1]-.5).mean())

    def test_first_order_meta_gradient(self):
        self.assertEqual(run_numpy_flow(self.root)["result"], "PASS")

    def test_real_research_manifest_counts(self):
        conf = configure("B0")
        manifest = load_manifest(conf)
        self.assertEqual(tuple(len(manifest[x]) for x in ("train", "val", "test")), (850, 150, 469))

    def test_no_implicit_flip_plus_ddpm(self):
        for name in ("S1", "S2", "S3", "S4", "S5"):
            conf = configure(name)
            self.assertEqual(conf["augment"], "none")
            self.assertNotEqual(conf["synthetic"]["mode"], "none")


if __name__ == "__main__":
    unittest.main()
