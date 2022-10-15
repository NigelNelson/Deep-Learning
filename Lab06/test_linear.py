from unittest import TestCase
import layers
import numpy as np
import torch
import unittest

class TestLinear(TestCase):
    """
    Responsible for testing the forward, backward, and step functions of the Linear class
    """
    def setUp(self):
        self.a = layers.Input((2,1), train=True)
        self.a.set(torch.tensor([[3],[5]],dtype=torch.float64))
        
        self.W1 = layers.Input((3,2), train=True)
        self.W1.set(torch.tensor([[6, 2],[5, 4], [6, 7]],dtype=torch.float64))
        
        self.b1 = layers.Input((3,1), train=True)
        self.b1.set(torch.tensor([[2],[3], [5]],dtype=torch.float64))
        
        self.x = layers.Input((3,1), train=True)
        self.x.set(torch.tensor([[3],[4],[5]],dtype=torch.float64))
        
        self.W2 = layers.Input((2,3), train=True)
        self.W2.set(torch.tensor([[2, -4, 1], [3, 7, 0]],dtype=torch.float64))
        
        self.b2 = layers.Input((2,1), train=True)
        self.b2.set(torch.tensor([[7], [-2]],dtype=torch.float64))
        
        self.linear1 = layers.Linear(self.a, self.W1, self.b1)
        self.linear2 = layers.Linear(self.x, self.W2, self.b2)
        self.linear2.accumulate_grad(torch.tensor([[-2], [3]],dtype=torch.float64))

    def test_forward(self):
        self.linear1.forward()
        np.testing.assert_allclose(self.linear1.output.numpy(),np.array([[30],[38],[58]]))   

    def test_backward(self):
        self.linear2.backward()
        np.testing.assert_allclose(self.linear2.x.grad.numpy(),np.array([[5],[29],[-2]]))
        np.testing.assert_allclose(self.linear2.W.grad.numpy(),np.array([[-6, -8, -10],[9, 12, 15]]))
        np.testing.assert_allclose(self.linear2.b.grad.numpy(),np.array([[-2],[3]]))

    def test_step(self):
        self.linear2.backward()
        self.linear2.x.step(0.1)
        self.linear2.W.step(0.1)
        self.linear2.b.step(0.1)
        np.testing.assert_allclose(self.linear2.x.output.numpy(),np.array([[2.5],[1.1],[5.2]]))
        np.testing.assert_allclose(self.linear2.W.output.numpy(),np.array([[2.6, -3.2, 2],[2.1, 5.8, -1.5]]))
        np.testing.assert_allclose(self.linear2.b.output.numpy(),np.array([[7.2],[-2.3]]))

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
