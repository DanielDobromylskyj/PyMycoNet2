from .. import Network, NetworkValidationException
from ..activation.relu import ReLU
from ..layers.dense import DenseLayer


def test_validate_layer_types():
    test_cases = [
        [(DenseLayer(2, 2), ReLU()), True],
        [(DenseLayer(7, 1),), True],
        [(DenseLayer(2, 2), ReLU), False],
        [("Hello!", ReLU()), False],
    ]

    for layout, should_work in test_cases:
        try:
            Network(layout, silent=True, validate=True)

            if not should_work:
                return [False, f"Validation missed 'illegal' layout -> {layout}"]
        except NetworkValidationException:
            if should_work:
                return [False, f"Validation flagged 'legal' network layout as 'illegal'-> {layout}"]

    return [True, None]
