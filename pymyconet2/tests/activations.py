from ..activation import relu, tanh, sigmoid, softmax

from .. import Network

def test_relu_forward_validation():
    test_cases = [
        (0.0, 0.0), (1.0, 1.0), (0.2, 0.2), (-0.2, 0.0), (-412.0, 0.0), (-4.0, 0.0), (-0.0, 0)
    ]

    basic_network = Network((
        relu.ReLU(),
    ), silent=True)

    for input_data, expected_output in test_cases:
        outputs = basic_network.forward_single([input_data])

        if round(outputs[0], 6) != expected_output:
            return [False, f"Target={expected_output}, Output={outputs}"]

    return [True, None]

def test_tanh_forward_validation():
    test_cases = [
        (0.0, 0.0),
        (5, 0.9999092),
        (100, 1.0),
        (-5, -0.9999092),
        (-100, -1.0)
    ]

    basic_network = Network((
        tanh.TanH(),
    ), silent=True)

    for input_data, expected_output in test_cases:
        outputs = basic_network.forward_single([input_data])

        if round(outputs[0], 6) != round(expected_output, 6):
            return [False, f"Target={expected_output}, Output={outputs}"]

    return [True, None]


def test_sigmoid_forward_validation():
    test_cases = [
        (0.0, 0.5),
        (5, 0.9933071),
        (100, 1.0),
        (-5, 0.0066929),
        (-100, 0.0),
        (0.4, 0.5986877)
    ]

    basic_network = Network((
        sigmoid.Sigmoid(),
    ), silent=True)

    for input_data, expected_output in test_cases:
        outputs = basic_network.forward_single([input_data])

        if round(outputs[0], 6) != round(expected_output, 6):
            return [False, f"Target={expected_output}, Output={outputs}"]

    return [True, None]

def test_softmax_forward_validation():
    test_cases = [
        ([0.0, 1.0], [0.269, 0.731]),
        ([-4.0, 4.0], [0.0, 1.0]),
        ([5.0, 4.0], [0.731, 0.269]),
        ([0.0, 0.0], [0.5, 0.5]),
    ]

    basic_network = Network((
        softmax.SoftMax(),
    ), silent=True)

    for input_data, expected_output in test_cases:
        outputs = basic_network.forward_single(input_data)

        if [round(x, 3) for x in outputs] != [round(x, 3) for x in expected_output]:
            return [False, f"Target={expected_output}, Output={outputs}"]

    return [True, None]

def test_generic_gradient_return_validation():
    test_cases = [
        ([0.0, 1.0], [0.269, 0.731]),
        ([-4.0, 4.0], [0.0, 1.0]),
        ([5.0, 4.0], [0.731, 0.269]),
        ([0.0, 0.0], [0.5, 0.5]),
    ]

    basic_network = Network((
        softmax.SoftMax(),
    ), silent=True)

    for input_data, expected_output in test_cases:
        outputs = basic_network.train_batched_epoche([input_data], [expected_output], 1)

        if outputs != [(None, None)]:
            return [False, f"Target={[(None, None)]}, Output={outputs}"]

    return [True, None]
