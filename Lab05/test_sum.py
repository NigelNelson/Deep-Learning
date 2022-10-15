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
        
        self.W2 = layers.Input((3,2), train=True)
        self.W2.set(torch.tensor([[-6, 2],[-5, 4], [4, 3]],dtype=torch.float64))
        
        self.b1 = layers.Input((2,1), train=True)
        self.b1.set(torch.tensor([[1],[2]],dtype=torch.float64))
        self.sum = layers.Sum(self.a, self.b1)
        
        self.b2 = layers.Input((3,1), train=True)
        self.b2.set(torch.tensor([[2],[3], [5]],dtype=torch.float64))
        
        self.linear = layers.Linear(self.a, self.W1, self.b2)
        self.relu = layers.ReLU(self.a, self.W2, self.b2)
        self.mse = layers.MSELoss(self.a, self.b1)
        self.regularization = layers.Regularization(self.a)
        self.softmax = layers.Softmax(self.b1, self.a)

    def test_forward(self):
        self.sum.forward()

    def test_backward(self):
        self.sum.forward()
        self.sum.accumulate_grad(torch.ones(2,1))
        self.sum.backward()

        np.testing.assert_allclose(self.a.grad.numpy(),np.ones((2,1)))
        np.testing.assert_allclose(self.b.grad.numpy(),np.ones((2,1)))

    def test_step(self):
        self.sum.forward()
        self.sum.accumulate_grad(np.ones((2,1)))
        self.sum.backward()
        self.a.step(step_size = 0.1)
        self.b.step(step_size = 0.1)

        np.testing.assert_allclose(self.a.output.numpy(),np.array([[2.9],[4.9]]))
        np.testing.assert_allclose(self.b.output.numpy(),np.array([[0.9],[1.9]]))
        
    def test_input_forward(self):
        self.a.forward()
        np.testing.assert_allclose(self.a.output.numpy(),np.array([[3],[5]]))   
        
    def test_linear_forward(self):
        self.linear.forward()
        np.testing.assert_allclose(self.linear.output.numpy(),np.array([[30],[38],[58]]))
        
    def test_relu_forward(self):
        self.relu.forward()
        np.testing.assert_allclose(self.relu.output.numpy(),np.array([[0],[8],[32]]))
        
    def test_mse_forward(self):
        self.mse.forward()
        np.testing.assert_allclose(self.mse.output.numpy(),np.array([6.5]))
        
    def test_reg_forward(self):
        self.regularization.forward()
        np.testing.assert_allclose(self.regularization.output.numpy(),np.array([34]))
        
    def test_softmax_forward(self):
        self.softmax.forward()
        np.testing.assert_allclose(self.softmax.classifications.numpy(),np.array([[0.11920292202], [0.88079707797]]))
        np.testing.assert_allclose(self.softmax.output.numpy(),np.array([2.38078403312]), rtol=1e-06)    

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
