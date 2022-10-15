from unittest import TestCase
import layers
import numpy as np
import torch
import unittest

class TestSum(TestCase):
    """
    Please note: I (Dr. Yoder) may have assumed different parameters for my network than you use.
    TODO: Update these tests to work with YOUR definitions of arguments and variables.
    """
    def setUp(self):
        self.a = layers.Input((2,1), train=True)
        self.a.set(torch.tensor([[3],[5]],dtype=torch.float64))
        
        self.b1 = layers.Input((2,1), train=True)
        self.b1.set(torch.tensor([[1],[2]],dtype=torch.float64))
        
        self.mse = layers.MSELoss(self.a, self.b1)

    def test_forward(self):
        self.mse.forward()
        np.testing.assert_allclose(self.mse.output.numpy(),np.array([6.5]))

    def test_backward(self):
        pass

    def test_step(self):
        pass

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
