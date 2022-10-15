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
        
        self.W1 = layers.Input((3,2), train=True)
        self.W1.set(torch.tensor([[6, 2],[5, 4], [6, 7]],dtype=torch.float64))
        
        self.b2 = layers.Input((3,1), train=True)
        self.b2.set(torch.tensor([[2],[3], [5]],dtype=torch.float64))
        
        self.linear = layers.Linear(self.a, self.W1, self.b2)

    def test_forward(self):
        self.linear.forward()
        np.testing.assert_allclose(self.linear.output.numpy(),np.array([[30],[38],[58]]))   

    def test_backward(self):
        pass

    def test_step(self):
        pass 

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
