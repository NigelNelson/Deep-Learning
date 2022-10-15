from unittest import TestCase
import layers
import numpy as np
import torch
import unittest

class TestRelu(TestCase):
    """
    Responsible for testing the forward, backward, and step functions of the Sum class
    """
    def setUp(self):
        self.a = layers.Input((3,1), train=True)
        self.a.set(torch.tensor([[-1],[2],[5]],dtype=torch.float64))
        
        self.relu = layers.ReLU(self.a)
        
        self.relu.accumulate_grad(torch.tensor([[-13], [9], [-2]],dtype=torch.float64))

    def test_forward(self):
        self.relu.forward()
        np.testing.assert_allclose(self.relu.output.numpy(),np.array([[0],[2],[5]]))

    def test_backward(self):
        self.relu.backward()
        np.testing.assert_allclose(self.relu.input_layer.grad.numpy(),np.array([[0],[9],[-2]]))

    def test_step(self):
        self.relu.backward()
        self.relu.input_layer.step(0.1)
        np.testing.assert_allclose(self.relu.input_layer.output.numpy(),np.array([[-1],[1.1],[5.2]]))

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
