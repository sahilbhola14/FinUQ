# Purpose: Class for Operand (Custom class for operands)
#          This class is used to keep track of the number of operations done on the operand
#          and the effort of the operand, i.e., the number of operations required to compute the operand
# Written by: Sahil Bhola, June 13, 2023

import numpy as np

class Operand:
    """Class for Operand
    ATTRIBUTES:
    val: value of the operand
    effort: effort of the operand, i.e., the number of operations required to compute the operand
    num_ops: Number of operations done on the operand (For node)
    """
    def __init__(self, val, effort=0):
        self.val = val
        self.effort = effort
        self.num_ops = 0

        # Check type of val
        self._check_type()

        # Check the shape
        self._check_shape()

        # Get the length of val
        self.length = self._get_length()

    def __repr__(self):
        return f"Operand(val: {self.val}, length: {self.length}, effort: {self.effort}, num_ops: {self.num_ops})"

    def _get_length(self):
        if isinstance(self.val, np.ndarray):
            return self.val.shape[0]
        elif isinstance(self.val, float) or isinstance(self.val, int):
            return 1
        else:
            raise ValueError("Invalid type of operand")

    def _check_shape(self):
        if isinstance(self.val, np.ndarray):
            assert len(self.val.shape) == 1, "Invalid shape of operand"
        elif isinstance(self.val, float) or isinstance(self.val, int):
            pass

    def _check_type(self):
        assert isinstance(self.val, (np.ndarray, float, int)), "Invalid type of operand"

    def _increase_num_ops(self, other):
        ops_factor = self._get_ops_factor(other)
        self.num_ops += ops_factor

    def _increase_effort(self, other):
        effort_factor = self._get_effort_factor(other)
        return self.effort + effort_factor

    def _get_effort(self):
        return self.effort

    def _get_effort_factor(self, other):
        condition = other.length == self.length or other.length == 1 or self.length == 1
        assert condition, "Inconsistent length of operands"

        if self.length == 1 and other.length != 1:
            return other.length
        elif other.length == 1 and self.length != 1:
            return self.length
        elif self.length == other.length == 1:
            return 1
        elif self.length == other.length:
            return self.length
        else:
            raise ValueError("Invalid length of operands")

    def _get_ops_factor(self, other):
        condition = other.length == self.length or other.length == 1 or self.length == 1
        assert condition, "Inconsistent length of operands"

        if self.length == 1 and other.length != 1:
            return other.length
        elif other.length == 1 and self.length != 1:
            return self.length
        elif self.length == other.length == 1:
            return 1
        elif self.length == other.length:
            return 2*self.length
        else:
            raise ValueError("Invalid length of operands")

    def _set_effort(self, effort):
        self.effort = effort

    def __add__(self, other):
        self._increase_num_ops(other)
        effort = self._increase_effort(other)
        return Operand(self.val + other.val, effort)

    def __sub__(self, other):
        self._increase_num_ops(other)
        effort = self._increase_effort(other)
        return Operand(self.val - other.val, effort)

    def __mul__(self, other):
        self._increase_num_ops(other)
        effort = self._increase_effort(other)
        return Operand(self.val * other.val, effort)

    def __truediv__(self, other):
        self._increase_num_ops(other)
        effort = self._increase_effort(other)
        return Operand(self.val / other.val, effort)

    def __pow__(self, other):
        self._increase_num_ops(other)
        effort = self._increase_effort(other)
        return Operand(self.val ** other.val, effort)

    def __mod__(self, other):
        self._increase_num_ops(other)
        effort = self._increase_effort(other)
        return Operand(self.val % other.val, effort)

    def __floordiv__(self, other):
        self._increase_num_ops(other)
        effort = self._increase_effort(other)
        return Operand(self.val // other.val, effort)

    def __sqrt__(self):
        self._increase_num_ops(Operand(1))
        effort = self._increase_effort(Operand(1))
        return Operand(np.sqrt(self.val), effort)

    def __exp__(self):
        self._increase_num_ops(Operand(1))
        effort = self._increase_effort(Operand(1))
        return Operand(np.exp(self.val), effort)

    def __and__(self, other):
        return self.val & other.val

    def __or__(self, other):
        return self.val | other.val

    def __xor__(self, other):
        return self.val ^ other.val

    def __lshift__(self, other):
        return self.val << other.val

    def __rshift__(self, other):
        return self.val >> other.val

    def __invert__(self):
        return ~self.val

    def __neg__(self):
        return -self.val

    def __pos__(self):
        return +self.val

    def __abs__(self):
        return abs(self.val)

    def __lt__(self, other):
        return self.val < other.val

    def __le__(self, other):
        return self.val <= other.val

    def __eq__(self, other):
        return self.val == other.val

    def __ne__(self, other):
        return self.val != other.val

    def __gt__(self, other):
        return self.val > other.val
