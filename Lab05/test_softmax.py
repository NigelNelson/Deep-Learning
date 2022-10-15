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

        self.softmax = layers.Softmax(self.b1, self.a)

    def test_forward(self):
        self.softmax.forward()
        np.testing.assert_allclose(self.softmax.classifications.numpy(),np.array([[0.11920292202], [0.88079707797]]))
        np.testing.assert_allclose(self.softmax.output.numpy(),np.array([2.38078403312]), rtol=1e-06)    

    def test_backward(self):
        pass

    def test_step(self):
        pass

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
