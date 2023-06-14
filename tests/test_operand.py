# Script to test the operand class
import numpy as np
from operand.operand import Operand as op

def test_scalar_addition():
    """Test the addition of two scalar operands"""
    a = op(1)
    b = op(2)
    c = a + b
    assert c.val == 3, "Addition failed"
    assert c.num_ops == 0, "Incorrect number of operations"
    assert c.effort == 1, "Incorrect effort"
    assert (isinstance(c, op)), "Incorrect type"
    assert (isinstance(c.val, int)), "Incorrect value type"

def test_scalar_subtraction():
    """Test the subtraction of two scalar operands"""
    a = op(1)
    b = op(2)
    c = a - b
    assert c.val == -1, "Subtraction failed"
    assert c.num_ops == 0, "Incorrect number of operations"
    assert c.effort == 1, "Incorrect effort"
    assert (isinstance(c, op)), "Incorrect type"
    assert (isinstance(c.val, int)), "Incorrect value type"

def test_scalar_multiplication():
    """Test the multiplication of two scalar operands"""
    a = op(1.5)
    b = op(2.0)
    c = a * b
    assert c.val == 3.0, "Multiplication failed"
    assert c.num_ops == 0, "Incorrect number of operations"
    assert c.effort == 1, "Incorrect effort"
    assert (isinstance(c, op)), "Incorrect type"
    assert (isinstance(c.val, float)), "Incorrect value type"

def test_scalar_division():
    """Test the division of two scalar operands"""
    a = op(1)
    b = op(2)
    c = a / b
    assert c.val == 0.5, "Division failed"
    assert c.num_ops == 0, "Incorrect number of operations"
    assert c.effort == 1, "Incorrect effort"
    assert (isinstance(c, op)), "Incorrect type"
    assert (isinstance(c.val, float)), "Incorrect value type"

def test_scalar_true_divide():
    """Test the true division of two scalar operands"""
    a = op(1.5)
    b = op(2.0)
    c = a / b
    assert c.val == 0.75, "True division failed"
    assert c.num_ops == 0, "Incorrect number of operations"
    assert c.effort == 1, "Incorrect effort"
    assert (isinstance(c, op)), "Incorrect type"
    assert (isinstance(c.val, float)), "Incorrect value type"

def test_scalar_power():
    """Test the power of two scalar operands"""
    a = op(1.5)
    b = op(2)
    c = a ** b
    assert c.val == 2.25, "Power failed"
    assert c.num_ops == 0, "Incorrect number of operations"
    assert c.effort == 1, "Incorrect effort"
    assert (isinstance(c, op)), "Incorrect type"
    assert (isinstance(c.val, float)), "Incorrect value type"

def test_scalar_modulo():
    """Test the modulo of two scalar operands"""
    a = op(1)
    b = op(2)
    c = a % b
    assert c.val == 1, "Modulo failed"
    assert c.num_ops == 0, "Incorrect number of operations"
    assert c.effort == 1, "Incorrect effort"
    assert (isinstance(c, op)), "Incorrect type"
    assert (isinstance(c.val, int)), "Incorrect value type"

def test_floor_divide():
    """Test the floor division of two scalar operands"""
    a = op(1)
    b = op(2)
    c = a // b
    assert c.val == 0, "Floor division failed"
    assert c.num_ops == 0, "Incorrect number of operations"
    assert c.effort == 1, "Incorrect effort"
    assert (isinstance(c, op)), "Incorrect type"
    assert (isinstance(c.val, int)), "Incorrect value type"

def test_scalar_square_root():
    """Test the square root of a scalar operand"""
    a = op(4)
    c = a.__sqrt__()
    assert c.val == 2, "Square root failed"
    assert c.num_ops == 0, "Incorrect number of operations"
    assert c.effort == 1, "Incorrect effort"
    assert (isinstance(c, op)), "Incorrect type"
    assert (isinstance(c.val, float)), "Incorrect value type"

def test_scalar_exponential():
    """Test the exponential of a scalar operand"""
    a = op(2)
    c = a.__exp__()
    assert (c.val - 7.38905609893065 < 1e-15), "Exponential failed"
    assert c.num_ops == 0, "Incorrect number of operations"
    assert c.effort == 1, "Incorrect effort"
    assert (isinstance(c, op)), "Incorrect type"
    assert (isinstance(c.val, float)), "Incorrect value type"

def test_scalar_vector_addition():
    """Test the addition of a scalar and a vector operand"""
    a = op(2)
    b = op(np.array([1, 2, 3]))
    c = a + b
    assert (np.linalg.norm(c.val - np.array([3, 4, 5])) < 1e-15), "Vector addition failed"
    assert c.num_ops == 0, "Incorrect number of operations"
    assert c.effort == 3, "Incorrect effort"
    assert isinstance(c, op), "Incorrect type"
    assert isinstance(c.val, np.ndarray), "Incorrect value type"

def test_scalar_vector_subtraction():
    """Test the subtraction of a scalar and a vector operand"""
    a = op(2)
    b = op(np.array([1, 2, 3]))
    c = a - b
    assert (np.linalg.norm(c.val - np.array([1, 0, -1])) < 1e-15), "Vector subtraction failed"
    assert c.num_ops == 0, "Incorrect number of operations"
    assert c.effort == 3, "Incorrect effort"
    assert isinstance(c, op), "Incorrect type"
    assert isinstance(c.val, np.ndarray), "Incorrect value type"

def test_scalar_vector_multiplication():
    """Test the multiplication of a scalar and a vector operand"""
    a = op(2)
    b = op(np.array([1, 2, 3]))
    c = a * b
    assert (np.linalg.norm(c.val - np.array([2, 4, 6])) < 1e-15), "Vector multiplication failed"
    assert c.num_ops == 0, "Incorrect number of operations"
    assert c.effort == 3, "Incorrect effort"
    assert isinstance(c, op), "Incorrect type"
    assert isinstance(c.val, np.ndarray), "Incorrect value type"

def test_scalar_vector_division():
    """Test the division of a scalar and a vector operand"""
    a = op(2)
    b = op(np.array([1, 2, 3]))
    c = a / b
    assert (np.linalg.norm(c.val - np.array([2, 1, 2/3])) < 1e-15), "Vector division failed"
    assert c.num_ops == 0, "Incorrect number of operations"
    assert c.effort == 3, "Incorrect effort"
    assert isinstance(c, op), "Incorrect type"
    assert isinstance(c.val, np.ndarray), "Incorrect value type"

def test_vector_vector_addition():
    """Test the addition of two vector operands"""
    a = op(np.array([1, 2, 3]))
    b = op(np.array([1, 2, 3]))
    c = a + b
    assert (np.linalg.norm(c.val - np.array([2, 4, 6])) < 1e-15), "Vector addition failed"
    assert c.num_ops == 0, "Incorrect number of operations"
    assert c.effort == 3, "Incorrect effort"
    assert isinstance(c, op), "Incorrect type"
    assert isinstance(c.val, np.ndarray), "Incorrect value type"

def test_vector_vector_subtraction():
    """Test the subtraction of two vector operands"""
    a = op(np.array([1, 2, 3]))
    b = op(np.array([1, 2, 3]))
    c = a - b
    assert (np.linalg.norm(c.val - np.array([0, 0, 0])) < 1e-15), "Vector subtraction failed"
    assert c.num_ops == 0, "Incorrect number of operations"
    assert c.effort == 3, "Incorrect effort"
    assert isinstance(c, op), "Incorrect type"
    assert isinstance(c.val, np.ndarray), "Incorrect value type"

def test_vector_vector_multiplication():
    """Test the multiplication of two vector operands"""
    a = op(np.array([1, 2, 3]))
    b = op(np.array([1, 2, 3]))
    c = a * b
    assert (np.linalg.norm(c.val - np.array([1, 4, 9])) < 1e-15), "Vector multiplication failed"
    assert c.num_ops == 0, "Incorrect number of operations"
    assert c.effort == 3, "Incorrect effort"
    assert isinstance(c, op), "Incorrect type"
    assert isinstance(c.val, np.ndarray), "Incorrect value type"

def test_vector_vector_division():
    """Test the division of two vector operands"""
    a = op(np.array([1, 2, 3]))
    b = op(np.array([1, 2, 3]))
    c = a / b
    assert (np.linalg.norm(c.val - np.array([1, 1, 1])) < 1e-15), "Vector division failed"
    assert c.num_ops == 0, "Incorrect number of operations"
    assert c.effort == 3, "Incorrect effort"
    assert isinstance(c, op), "Incorrect type"
    assert isinstance(c.val, np.ndarray), "Incorrect value type"

def test_scalar_addition_effort():
    """Test the effort of the addition of a scalar and a scalar operand"""
    a = op(2)
    b = op(3)
    c = a + b
    d = c + b
    assert a.effort == 0, "Incorrect effort"
    assert b.effort == 0, "Incorrect effort"
    assert c.effort == 1, "Incorrect effort"
    assert d.effort == 2, "Incorrect effort"

def test_scalar_subtraction_effort():
    """Test the effort of the subtraction of a scalar and a scalar operand"""
    a = op(2)
    b = op(3)
    c = a - b
    d = c - b
    assert a.effort == 0, "Incorrect effort"
    assert b.effort == 0, "Incorrect effort"
    assert c.effort == 1, "Incorrect effort"
    assert d.effort == 2, "Incorrect effort"

def test_scalar_multiplication_effort():
    """Test the effort of the multiplication of a scalar and a scalar operand"""
    a = op(2)
    b = op(3)
    c = a * b
    d = c * b
    assert a.effort == 0, "Incorrect effort"
    assert b.effort == 0, "Incorrect effort"
    assert c.effort == 1, "Incorrect effort"
    assert d.effort == 2, "Incorrect effort"

def test_scalar_division_effort():
    """Test the effort of the division of a scalar and a scalar operand"""
    a = op(2)
    b = op(3)
    c = a / b
    d = c / b
    assert a.effort == 0, "Incorrect effort"
    assert b.effort == 0, "Incorrect effort"
    assert c.effort == 1, "Incorrect effort"
    assert d.effort == 2, "Incorrect effort"

def test_vector_addition_effort():
    """Test the effort of the addition of a vector and a vector operand"""
    a = op(np.array([1, 2, 3]))
    b = op(np.array([1, 2, 3]))
    c = a + b
    d = c + b
    assert a.effort == 0, "Incorrect effort"
    assert b.effort == 0, "Incorrect effort"
    assert c.effort == 3, "Incorrect effort"
    assert d.effort == 6, "Incorrect effort"

def test_vector_subtraction_effort():
    """Test the effort of the subtraction of a vector and a vector operand"""
    a = op(np.array([1, 2, 3]))
    b = op(np.array([1, 2, 3]))
    c = a - b
    d = c - b
    assert a.effort == 0, "Incorrect effort"
    assert b.effort == 0, "Incorrect effort"
    assert c.effort == 3, "Incorrect effort"
    assert d.effort == 6, "Incorrect effort"

def test_vector_multiplication_effort():
    """Test the effort of the multiplication of a vector and a vector operand"""
    a = op(np.array([1, 2, 3]))
    b = op(np.array([1, 2, 3]))
    c = a * b
    d = c * b
    assert a.effort == 0, "Incorrect effort"
    assert b.effort == 0, "Incorrect effort"
    assert c.effort == 3, "Incorrect effort"
    assert d.effort == 6, "Incorrect effort"

def test_vector_division_effort():
    """Test the effort of the division of a vector and a vector operand"""
    a = op(np.array([1, 2, 3]))
    b = op(np.array([1, 2, 3]))
    c = a / b
    d = c / b
    assert a.effort == 0, "Incorrect effort"
    assert b.effort == 0, "Incorrect effort"
    assert c.effort == 3, "Incorrect effort"
    assert d.effort == 6, "Incorrect effort"

def test_scalar_vector_addition_effort():
    """Test the effort of the addition of a scalar and a vector operand"""
    a = op(2)
    b = op(np.array([1, 2, 3]))
    c = a + b
    d = c + b
    assert a.effort == 0, "Incorrect effort"
    assert b.effort == 0, "Incorrect effort"
    assert c.effort == 3, "Incorrect effort"
    assert d.effort == 6, "Incorrect effort"

def test_scalar_vector_subtraction_effort():
    """Test the effort of the subtraction of a scalar and a vector operand"""
    a = op(2)
    b = op(np.array([1, 2, 3]))
    c = a - b
    d = c - b
    assert a.effort == 0, "Incorrect effort"
    assert b.effort == 0, "Incorrect effort"
    assert c.effort == 3, "Incorrect effort"
    assert d.effort == 6, "Incorrect effort"

def test_scalar_vector_multiplication_effort():
    """Test the effort of the multiplication of a scalar and a vector operand"""
    a = op(2)
    b = op(np.array([1, 2, 3]))
    c = a * b
    d = c * b
    assert a.effort == 0, "Incorrect effort"
    assert b.effort == 0, "Incorrect effort"
    assert c.effort == 3, "Incorrect effort"
    assert d.effort == 6, "Incorrect effort"

def test_scalar_vector_division_effort():
    """Test the effort of the division of a scalar and a vector operand"""
    a = op(2)
    b = op(np.array([1, 2, 3]))
    c = a / b
    d = c / b
    assert a.effort == 0, "Incorrect effort"
    assert b.effort == 0, "Incorrect effort"
    assert c.effort == 3, "Incorrect effort"
    assert d.effort == 6, "Incorrect effort"

def test_scalar_loop_addition():
    """Test the addition of a scalar and a loop operand
    NOTE:
        Each in place operation returns a new instance
    """
    a = op(2)
    for ii in range(2):
        a = a + op(1)
    assert a.val == 4, "Incorrect value"
    assert a.num_ops == 0, "Incorrect number of operations"
    assert a.effort == 2, "Incorrect effort"

def test_scalar_loop_subtraction():
    """Test the subtraction of a scalar and a loop operand
    NOTE:
        Each in place operation returns a new instance
    """
    a = op(10)
    for ii in range(2):
        a = a - op(1)
    assert a.val == 8, "Incorrect value"
    assert a.num_ops == 0, "Incorrect number of operations"
    assert a.effort == 2, "Incorrect effort"

def test_scalar_loop_multiplication():
    """Test the multiplication of a scalar and a loop operand
    NOTE:
        Each in place operation returns a new instance
    """
    a = op(2)
    for ii in range(2):
        a = a * op(2)

    assert a.val == 8, "Incorrect value"
    assert a.num_ops == 0, "Incorrect number of operations"
    assert a.effort == 2, "Incorrect effort"

def test_scalar_loop_division():
    """Test of the division of a scalar and a loop operand"""
    a = op(10)
    for ii in range(2):
        a = a / op(2)

    assert a.val == 2.5, "Incorrect value"
    assert a.num_ops == 0, "Incorrect number of operations"
    assert a.effort == 2, "Incorrect effort"

def test_get_length():
    """Test the length of the operand"""
    a = op(np.array([1, 2, 3]))
    assert a._get_length() == 3, "Incorrect length"
