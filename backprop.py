"""Code for HW3 Problem 1: Backpropagation."""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit as sigmoid

OPTS = None

EPSILON = 1e-8  # epsilon to use for numerical gradient computation

# A linearly separable classification dataset
SIMPLE_DATASET = [
    (np.array([2.0, 2.0]), 1),
    (np.array([1.0, 0.0]), 1),
    (np.array([1.0, 2.0]), -1),
    (np.array([0.0, -1.0]), -1),
]

# The XOR dataset, which is not linearly separable
XOR_DATASET = [
    (np.array([0.0, 0.0]), -1),
    (np.array([1.0, 1.0]), -1),
    (np.array([1.0, 0.0]), 1),
    (np.array([0.0, 1.0]), 1),
]

### Start of (modified) backprop lecture demo code ###


class Node(object):
    """Node in a computation graph.

    Must store the following:
        - self.value: Numerical value, computed during forward pass
        - self.grad: Gradient, computed during backward pass
        - self.topo_order: Topologically-sorted list of nodes
    """

    def backward(self):
        """Does backward pass.

        Assumption: self.grad has been set by parent nodes already.
        Propagates this gradient to all child nodes.
        """
        raise NotImplementedError

    def compute_grad(self):
        # This function only gets called on the output node.
        self.grad = np.float64(1)   # Gradient of output w.r.t itself is 1
        for node in self.topo_order[::-1]:  # Reverse the list
            # Call backward() in reverse topological order
            node.backward()


class InputNode(Node):
    def __init__(self, value, topo_order):
        self.value = value
        if isinstance(value, np.ndarray):
            # Initialize self.grad to an array of all zeros with the same shape as value
            self.grad = np.zeros_like(value, dtype=np.float64)
        else:
            self.grad = np.float64(0)
        self.topo_order = topo_order
        topo_order.append(self)

    def backward(self):
        pass  # Nothing to do, as there are no child nodes


class AddNode(Node):
    def __init__(self, n1, n2):
        self.n1 = n1
        self.n2 = n2
        self.value = n1.value + n2.value
        if isinstance(self.value, np.ndarray):
            # Initialize self.grad to an array of all zeros with the same shape as value
            self.grad = np.zeros_like(self.value, dtype=np.float64)
        else:
            self.grad = np.float64(0)
        self.topo_order = n1.topo_order
        self.topo_order.append(self)

    def backward(self):
        # Recursively, self.grad has gradient of output w.r.t. this node
        # Changing n1 changes (n1 + n2) by the same amount
        # Same for n2
        # Thus gradient w.r.t. n1 and n2 is the same as gradient w.r.t this node
        self.n1.grad += self.grad
        self.n2.grad += self.grad


class MulNode(Node):
    def __init__(self, n1, n2):
        self.n1 = n1
        self.n2 = n2
        self.value = n1.value * n2.value
        self.grad = np.float64(0)
        self.topo_order = n1.topo_order
        self.topo_order.append(self)

    def backward(self):
        self.n1.grad += self.grad * self.n2.value
        self.n2.grad += self.grad * self.n1.value


class PowerNode(Node):
    def __init__(self, n, power):
        # n is a Node, power is just a float/integer
        self.n = n
        self.power = power
        self.value = n.value**power
        self.grad = np.float64(0)
        self.topo_order = n.topo_order
        self.topo_order.append(self)

    def backward(self):
        self.n.grad += self.power * self.n.value**(self.power-1) * self.grad


class ReluNode(Node):
    def __init__(self, n):
        self.n = n
        self.value = np.maximum(n.value, 0)
        self.topo_order = n.topo_order
        self.topo_order.append(self)
        if isinstance(self.value, np.ndarray):
            # Initialize self.grad to an array of all zeros with the same shape as value
            self.grad = np.zeros_like(self.value, dtype=np.float64)
        else:
            self.grad = np.float64(0)

    def backward(self):
        self.n.grad += self.grad * (self.value > 0)

### End of Backprop lecture demo code ###


class ConstantMulNode(Node):
    """Node that computes a * x for some constant a."""

    def __init__(self, n, a):
        self.n = n
        self.a = a
        self.value = a * n.value
        self.grad = np.float64(0)
        self.topo_order = n.topo_order
        self.topo_order.append(self)

    def backward(self):
        self.n.grad += self.grad * self.a


class LogNode(Node):
    """Node that computes log(x), where x is the value of node n."""

    def __init__(self, n):
        self.n = n
        self.topo_order = n.topo_order
        self.topo_order.append(self)
        # TODO: Set self.value (forward pass) and initialize self.grad
        # BEGIN_SOLUTION 1b
        if np.any(n.value) <= 0:
            raise ValueError
        self.value = np.log(n.value)
        self.grad = np.float64(0)
        # END_SOLUTION 1b

    def backward(self):
        # BEGIN_SOLUTION 1b
        self.n.grad += self.grad / self.n.value
        # END_SOLUTION 1b


class SigmoidNode(Node):
    """Node that computes sigmoid(x), where x is the value of node n."""

    def __init__(self, n):
        self.n = n
        self.topo_order = n.topo_order
        self.topo_order.append(self)
        # TODO: Set self.value (forward pass) and initialize self.grad
        # BEGIN_SOLUTION 1e
        self.value = sigmoid(n.value)
        self.grad = np.float64(0)
        # END_SOLUTION 1e

    def backward(self):
        # BEGIN_SOLUTION 1e
        self.n.grad += self.grad * self.value * (1 - self.value)
        # END_SOLUTION 1e


class DotNode(Node):
    """Node that computes x^T v.

    n1 is a node whose value is a vector x.
    n2 is a node whose value is a vector v.
    """

    def __init__(self, n1, n2):
        self.n1 = n1
        self.n2 = n2
        self.topo_order = n1.topo_order
        self.topo_order.append(self)
        # TODO: Set self.value (forward pass) and initialize self.grad
        # BEGIN_SOLUTION 1g
        self.value = np.dot(n1.value, n2.value)
        self.grad = np.float64(0)
        # END_SOLUTION 1g

    def backward(self):
        # BEGIN_SOLUTION 1g
        self.n1.grad += self.grad * self.n2.value
        self.n2.grad += self.grad * self.n1.value
        # END_SOLUTION 1g


class MVMulNode(Node):
    """Node that computes the matrix-vector product Mv.

    M is a node whose value is a matrix of shape (p, d).
    v is a node whose value is a vector of length d.
    """

    def __init__(self, M, v):
        p, d = M.value.shape
        self.M = M  # p x d
        self.v = v  # d
        self.topo_order = M.topo_order
        self.topo_order.append(self)
        # TODO: Set self.value (forward pass) and initialize self.grad
        # BEGIN_SOLUTION 1k
        self.value = np.matmul(M.value, v.value)
        self.grad = np.zeros_like(self.value)
        # END_SOLUTION 1k

    def backward(self):
        # BEGIN_SOLUTION 1k
        # import pdb
        # pdb.set_trace()
        self.v.grad += np.matmul(self.M.value.T, self.grad)
        # self.v.grad = self.grad

        self.M.grad += np.outer(self.grad, self.v.value)
        # END_SOLUTION 1k


def gradient_check_1(a_test, b_test, c_test):
    """Run gradient check for LogNode, SigmoidNode, and DotNode."""
    # First let's define a helper function.
    # Given numpy arrays a_val, b_val, and c_val, this function:
    #     - Creates a computation graph that computes f(a_val, b_val, c_val)
    #     - Returns the output node of this computation graph
    def compute_f(a_val, b_val, c_val):
        """Helper function for gradient checking.

        Args: numpy array values for a, b, and c
        Returns: 2-tuple containing:
            - input_nodes: list of InputNode objects corresponding to a, b, and c
            - output_node: The output node whose value is f(a_val, b_val, c_val)
        """
        # Create the shared topo_order list and InputNode's
        topo_order = []
        a = InputNode(a_val, topo_order)
        b = InputNode(b_val, topo_order)
        c = InputNode(c_val, topo_order)
        input_nodes = [a, b, c]

        # Now, create the computation graph that computes f
        output_node = None  # TODO: Set this so we can return it
        # BEGIN_SOLUTION 1h
        d1 = DotNode(a, b)
        d2 = DotNode(a, c)
        l = LogNode(d1)
        s = SigmoidNode(d2)
        e = ConstantMulNode(s, -3)
        output_node = AddNode(l, e)
        # END_SOLUTION 1h
        return input_nodes, output_node

    # Compute gradients with backpropagation
    (a, b, c), out = compute_f(a_test, b_test, c_test)
    print(f'Computed value of f(a_test, b_test, c_test): {out.value}')
    out.compute_grad()  # Runs backpropagation, setting a.grad, b.grad, and c.grad
    # Now, numerically compute the gradient by changing each coordinate of each input by EPSILON
    # and observing how much f changes.
    # TODO: store the numerical gradients in the variables below.
    # Numerically computed gradient with respect to a
    a_grad_numerical = np.zeros(2)
    # Numerically computed gradient with respect to b
    b_grad_numerical = np.zeros(2)
    # Numerically computed gradient with respect to c
    c_grad_numerical = np.zeros(2)
    # BEGIN_SOLUTION 1h
    modify_first_dim = np.array([EPSILON, 0])
    modify_second_dim = np.array([0, EPSILON])
    a_grad_numerical[0] = (compute_f(
        a.value + modify_first_dim, b.value, c.value)[1].value - out.value) / EPSILON
    a_grad_numerical[1] = (compute_f(
        a.value + modify_second_dim, b.value, c.value)[1].value - out.value) / EPSILON
    b_grad_numerical[0] = (compute_f(
        a.value, b.value + modify_first_dim, c.value)[1].value - out.value) / EPSILON
    b_grad_numerical[1] = (compute_f(
        a.value, b.value + modify_second_dim, c.value)[1].value - out.value) / EPSILON
    c_grad_numerical[0] = (compute_f(
        a.value, b.value, c.value + modify_first_dim)[1].value - out.value) / EPSILON
    c_grad_numerical[1] = (compute_f(
        a.value, b.value, c.value + modify_second_dim)[1].value - out.value) / EPSILON
    # END_SOLUTION 1h

    # Finally, this code checks the difference between the gradients computed by backprop
    # and the gradients computed numerically.
    # The max difference should be smaller than 1e-7
    print('a.grad:')
    print(f'    Computed numerically  : {a_grad_numerical}')
    print(f'    Computed with backprop: {a.grad}')
    print(f'    Max difference: {np.max(np.abs(a_grad_numerical - a.grad))}')
    print('b.grad:')
    print(f'    Computed numerically  : {b_grad_numerical}')
    print(f'    Computed with backprop: {b.grad}')
    print(f'    Max difference: {np.max(np.abs(b_grad_numerical - b.grad))}')
    print('c.grad:')
    print(f'    Computed numerically  : {c_grad_numerical}')
    print(f'    Computed with backprop: {c.grad}')
    print(f'    Max difference: {np.max(np.abs(c_grad_numerical - c.grad))}')


def make_logistic_regression(dataset):
    """Create a function that computes the logistic regression loss.

    Args:
        dataset: A list of (x, y) pairs where x is a numpy array and y is in {-1, 1}
    Returns:
        compute_loss: A function that takes in numpy values w_val and b_val, and returns:
            - A tuple (w, b) of InputNode objects for w and b, respectively
            - total_loss, a Node whose value is the logistic regression loss for w and b on the given dataset
    """
    def compute_loss(w_val, b_val):
        topo_order = []
        w = InputNode(w_val, topo_order)
        b = InputNode(b_val, topo_order)
        # TODO: Set total_loss to be a Node whose value is the logistic regression loss
        total_loss = None
        # BEGIN_SOLUTION 1
        total_loss = InputNode(0, [])
        for item in dataset:
            x, y = item  # x ndarray
            # d is a reference to this specific DotNode
            d = DotNode(w, InputNode(x, []))
            z = AddNode(d, b)
            m = MulNode(z, InputNode(y, []))
            a = SigmoidNode(m)
            l = LogNode(a)
            o = ConstantMulNode(l, -1)
            total_loss = AddNode(o, total_loss)

        # END_SOLUTION 1i
        return (w, b), total_loss

    return compute_loss


def gradient_descent(f, params, lr, num_iters):
    """Run gradient descent on a given function.

    Args:
        - f: A function that takes in a list of parameter values and returns
            - param_nodes, an iterable of InputNode objects for each parameter
            - loss, a Node object whose value is the loss being optimized
        - params: A list of initial parameter values as numpy objects.
            This list will get updated with the values chosen by gradient descent
        - lr: Learning rate for gradient descent
        - num_iters: Number of iterations for gradient descent
    """
    for t in range(num_iters):
        # Run the loss function f with the current parameters
        input_nodes, loss = f(*params)
        # input_nodes is a tuple containing the InputNode for each parameter in params
        if t % 100 == 0:
            print(f't={t}: loss={loss.value}')

        # TODO: Update each parameter in params via gradient descent
        # BEGIN_SOLUTION 1i
        loss.compute_grad()
        for i, param in enumerate(params):
            params[i] -= lr*input_nodes[i].grad
#        params = [param-lr*input_nodes[i].grad for i,param in enumerate(params)]
        # END_SOLUTION 1i
    print(f'Final loss={loss.value}')


def gradient_check_2(M_test, v_test):
    """Run gradient check for MVMulNode."""
    # First let's define a helper function.
    # Given numpy arrays M_val and v_val, this function:
    #     - Creates a computation graph that computes f(M_val, v_val)
    #     - Returns the output node of this computation graph
    def compute_f(M_val, v_val):
        """Helper function for gradient checking.

        Args: numpy array values for M and c
        Returns: 2-tuple containing:
            - input_nodes: list of InputNode objects corresponding to M and v
            - output_node: The output node whose value is f(M, v)
        """
        # Create the shared topo_order list and InputNode's
        topo_order = []
        M = InputNode(M_val, topo_order)
        v = InputNode(v_val, topo_order)
        input_nodes = [M, v]

        # Now, create the computation graph that computes f
        output_node = None  # TODO: Set this so we can return it
        # BEGIN_SOLUTION 1m
        t1 = MVMulNode(M, v)
        t2 = ReluNode(t1)
        v.value = v.value.T
        output_node = DotNode(v, t2)

        # END_SOLUTION 1m
        return input_nodes, output_node

    # Compute gradients with backpropagation
    (M, v), out = compute_f(M_test, v_test)
    print(f'Computed value of f(M_test, v_test): {out.value}')
    out.compute_grad()  # Runs backpropagation, setting M.grad and v.grad

    # Now, numerically compute the gradient by changing each coordinate of each input by EPSILON
    # and observing how much f changes.
    # TODO: store the numerical gradients in the variables below.
    # Numerically computed gradient with respect to M
    M_grad_numerical = np.zeros((2, 2))
    # Numerically computed gradient with respect to v
    v_grad_numerical = np.zeros(2)
    # BEGIN_SOLUTION 1m

    M_grad_numerical[0, 0] = (compute_f(
        M.value + np.array([[EPSILON, 0], [0, 0]]), v.value)[1].value - out.value) / EPSILON
    M_grad_numerical[0, 1] = (compute_f(
        M.value + np.array([[0, EPSILON], [0, 0]]), v.value)[1].value - out.value) / EPSILON
    M_grad_numerical[1, 0] = (compute_f(
        M.value + np.array([[0, 0], [EPSILON, 0]]), v.value)[1].value - out.value) / EPSILON
    M_grad_numerical[1, 1] = (compute_f(
        M.value + np.array([[0, 0], [0, EPSILON]]), v.value)[1].value - out.value) / EPSILON
    v_grad_numerical[0] = (compute_f(
        M.value, v.value + np.array([EPSILON, 0]))[1].value - out.value) / EPSILON
    v_grad_numerical[1] = (compute_f(
        M.value, v.value + np.array([0, EPSILON]))[1].value - out.value) / EPSILON

    # END_SOLUTION 1m

    # Finally, this code checks the difference between the gradients computed by backprop
    # and the gradients computed numerically.
    # The max difference should be smaller than 1e-7
    print('M.grad:')
    print(f'    Computed numerically  : {M_grad_numerical}')
    print(f'    Computed with backprop: {M.grad}')
    print(f'    Max difference: {np.max(np.abs(M_grad_numerical - M.grad))}')
    print('v.grad:')
    print(f'    Computed numerically  : {v_grad_numerical}')
    print(f'    Computed with backprop: {v.grad}')
    print(f'    Max difference: {np.max(np.abs(v_grad_numerical - v.grad))}')


def make_neural_network(dataset):
    def compute_loss(W_val, b_val, v_val, c_val):
        topo_order = []
        W = InputNode(W_val, topo_order)
        b = InputNode(b_val, topo_order)
        v = InputNode(v_val, topo_order)
        c = InputNode(c_val, topo_order)
        # TODO: Set total_loss to be a Node whose value is the loss for the neural network
        total_loss = None
        # BEGIN_SOLUTION 1n
        total_loss = InputNode(0, topo_order)
        for pair in dataset:
            x,y = pair
            m = MVMulNode(W, InputNode(x, topo_order ))
            z = AddNode(m, b)
            a = ReluNode(z)
            k = DotNode(v, a)
            l = AddNode(k, c)
            n = ConstantMulNode(l, y)
            p = SigmoidNode(n)
            s = LogNode(p)
            o = ConstantMulNode(s, -1)
            total_loss = AddNode(o, total_loss)

            
        # END_SOLUTION 1n
        return (W, b, v, c), total_loss

    return compute_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'mode', choices=['grad_check_1', 'grad_check_2', 'logreg', 'neural'])
    parser.add_argument('--dataset', '-d', choices=['simple', 'xor'])
    return parser.parse_args()


def main():
    if OPTS.mode == 'grad_check_1':
        a_test = np.array([3.0, 2.0])
        b_test = np.array([-2.0, 7.0])
        c_test = np.array([1.0, -4.0])
        gradient_check_1(a_test, b_test, c_test)
    if OPTS.mode == 'grad_check_2':
        M_test = np.array([[2.0, -1.0], [0.0, 3.0]])
        v_test = np.array([1.0, -3.0])
        gradient_check_2(M_test, v_test)
    else:
        if OPTS.dataset == 'simple':
            dataset = SIMPLE_DATASET
        elif OPTS.dataset == 'xor':
            dataset = XOR_DATASET
        if OPTS.mode == 'logreg':
            loss_func = make_logistic_regression(dataset)
            # Initial values of w and b
            params = [np.array([0.0, 0.0]), np.float64(0)]
            gradient_descent(loss_func, params, 1.0, 1000)
            print(f'Learned w: {params[0]}')
            print(f'Learned b: {params[1]}')
        elif OPTS.mode == 'neural':
            loss_func = make_neural_network(dataset)
            # Initial values of W, b, v, and c
            # We initialize with small random numbers to break symmetry
            params = [np.array([[-.01, .07], [.17, 0], [-.04, -.11]]),
                      np.array([-.02, -.14, .05]),
                      np.array([.05, .14, .12]),
                      np.float64(-.02)]
            gradient_descent(loss_func, params, 0.1, 1000)
            print(f'Learned W: {params[0]}')
            print(f'Learned b: {params[1]}')
            print(f'Learned v: {params[2]}')
            print(f'Learned c: {params[3]}')


if __name__ == '__main__':
    OPTS = parse_args()
    main()
