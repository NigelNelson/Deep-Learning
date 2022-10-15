from unittest import TestCase
import layers
import numpy as np
import torch
import unittest

class TestMSE(TestCase):
    """
    Responsible for testing the forward, backward, and step functions of the MSE class
    """
    def setUp(self):
        self.a = layers.Input((2,1), train=True)
        self.a.set(torch.tensor([[3],[5]],dtype=torch.float64))
        
        self.b1 = layers.Input((2,1), train=True)
        self.b1.set(torch.tensor([[1],[2]],dtype=torch.float64))
        
        self.mse = layers.MSELoss(self.b1, self.a)

    def test_forward(self):
        self.mse.forward()
        np.testing.assert_allclose(self.mse.output.numpy(),np.array([6.5]))

    def test_backward(self):
        self.mse.backward()
        np.testing.assert_allclose(self.mse.predicted.grad.numpy(),np.array([[2],[3]]))

    def test_step(self):
        self.mse.backward()
        self.mse.predicted.step(0.1)
        np.testing.assert_allclose(self.mse.predicted.output.numpy(),np.array([[2.8],[4.7]]))

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
