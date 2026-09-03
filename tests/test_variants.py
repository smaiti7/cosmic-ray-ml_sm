import unittest

import torch

from cosmic_ml.model import build_model, model_config


class ModelVariantTest(unittest.TestCase):
    def test_both_variants_forward_and_report_type(self) -> None:
        for model_type in ("original", "attention"):
            with self.subTest(model_type=model_type):
                torch.manual_seed(2)
                model = build_model(model_type, torch.randn(7, 2))
                waveforms = torch.randn(3, 7, 48)
                mask = torch.ones(3, 7)
                mask[0, -1] = 0
                predictions = model(waveforms, mask)
                self.assertEqual(predictions["class_logit"].shape, (3,))
                self.assertEqual(predictions["energy"].shape, (3,))
                self.assertEqual(predictions["core"].shape, (3, 2))
                self.assertEqual(model_config(model)["model_type"], model_type)


if __name__ == "__main__":
    unittest.main()
