import unittest

import torch

from cosmic_ml.engine import multitask_loss
from cosmic_ml.model import TemporalDeepSet


class TemporalDeepSetTest(unittest.TestCase):
    def test_forward_and_backward(self) -> None:
        torch.manual_seed(1)
        model = TemporalDeepSet(torch.randn(7, 2), predict_direction=True)
        waveforms = torch.randn(3, 7, 48)
        mask = torch.ones(3, 7)
        mask[0, -1] = 0
        predictions = model(waveforms, mask)
        self.assertEqual(predictions["class_logit"].shape, (3,))
        self.assertEqual(predictions["energy"].shape, (3,))
        self.assertEqual(predictions["core"].shape, (3, 2))
        self.assertEqual(predictions["direction"].shape, (3, 3))
        torch.testing.assert_close(
            predictions["direction"].norm(dim=1), torch.ones(3), rtol=1e-5, atol=1e-5
        )

        batch = {
            "y_class": torch.tensor([0.0, 1.0, 1.0]),
            "y_energy": torch.tensor([0.0, -0.5, 0.4]),
            "y_core": torch.tensor([[0.0, 0.0], [0.2, -0.1], [0.4, 0.7]]),
            "y_direction": torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
        }
        loss, parts = multitask_loss(predictions, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(parts["loss"], 0)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()

