from unittest import TestCase
import layers
import numpy as np
import torch
import unittest

class TestSoftMax(TestCase):
    """
    Responsible for testing the forward, backward, and step functions of the SoftMax class
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
        self.softmax.backward()
        np.testing.assert_allclose(self.softmax.predicted.grad.numpy(),np.array([[2], [3]]))
        
    def test_step(self):
        self.softmax.backward()
        self.softmax.predicted.step(0.1)
        np.testing.assert_allclose(self.softmax.predicted.output.numpy(),np.array([[2.8], [4.7]]))

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
