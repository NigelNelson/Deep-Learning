from unittest import TestCase
import layers
import numpy as np
import torch
import unittest

class TestRegularization(TestCase):
    """
    Responsible for testing the forward, backward, and step functions of the Regularization class
    """
    def setUp(self):
        
        self.a = layers.Input((2,3), train=True)
        self.a.set(torch.tensor([[-2, 4, 1], [3, 7, 0]],dtype=torch.float64))
        
        self.regularization = layers.Regularization(self.a)
        
        self.regularization.accumulate_grad(torch.tensor([[1]],dtype=torch.float64))

    def test_forward(self):
        self.regularization.forward()
        np.testing.assert_allclose(self.regularization.output.numpy(),np.array([79])) 

    def test_backward(self):
        self.regularization.backward()
        np.testing.assert_allclose(self.regularization.input_tensor.grad.numpy(),np.array([[-4,8,2], [6,14,0]])) 

    def test_step(self):
        self.regularization.backward()
        self.regularization.input_tensor.step(0.1)
        np.testing.assert_allclose(self.regularization.input_tensor.output.numpy(),np.array([[-1.6,3.2,0.8], [2.4,5.6,0]])) 

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
