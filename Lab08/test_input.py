from unittest import TestCase
import layers
import numpy as np
import torch
import unittest

class TestInput(TestCase):
    """
    Responsible for testing the forward, backward, and step functions of the Input class
    """
    def setUp(self):
        self.a = layers.Input((2,1), train=True)
        self.a.set(torch.tensor([[3],[5]],dtype=torch.float64))
        self.a.accumulate_grad(torch.tensor([[1], [1]]))

    def test_forward(self):
        self.a.forward()
        np.testing.assert_allclose(self.a.output.numpy(),np.array([[3],[5]]))

    def test_backward(self):
        self.a.backward()
        np.testing.assert_allclose(self.a.grad.numpy(),np.array([[1], [1]]))
    
    def test_step(self):
        self.a.step(0.1)
        np.testing.assert_allclose(self.a.output.numpy(),np.array([[2.9], [4.9]]))


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
