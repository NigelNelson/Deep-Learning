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
        
        self.W2 = layers.Input((3,2), train=True)
        self.W2.set(torch.tensor([[-6, 2],[-5, 4], [4, 3]],dtype=torch.float64))
        
        self.b2 = layers.Input((3,1), train=True)
        self.b2.set(torch.tensor([[2],[3], [5]],dtype=torch.float64))
        
        self.relu = layers.ReLU(self.a, self.W2, self.b2)

    def test_forward(self):
        self.relu.forward()
        np.testing.assert_allclose(self.relu.output.numpy(),np.array([[0],[8],[32]]))

    def test_backward(self):
        pass

    def test_step(self):
        pass

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
