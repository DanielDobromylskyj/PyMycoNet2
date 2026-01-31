import numpy as np

from ..util.layer_data import LayerData, Gradients
from ..util.kernel import Kernel


class LayerInitializationException(Exception):
    pass


def required_init(func):
    def wrapper(self, *args, **kwargs):
        if not self.initialised:
            raise RuntimeError("Layer not initialised")
        return func(self, *args, **kwargs)
    return wrapper


class DefaultLayer:
    def __init__(self, kernels: list[Kernel], is_loading=False):
        self.kernels = kernels
        self.ctx = None
        self.queue = None
        self.initialised = False

        self.weights = None
        self.biases = None
        self.is_activation = False
        self.is_transform = False
        self.is_loading = is_loading

    def get_kernel(self, path: str):
        for kernel in self.kernels:
            if kernel.path == path:
                return kernel

        return None

    def _init(self): # Used for layer specific initialization - Must set weights / biases
        raise NotImplementedError("No Init function for Layer defined")
    
    def init(self, ctx, queue):
        self.ctx = ctx
        self.queue = queue

        self._init()

        if not self.is_loading and (self.weights is None or self.biases is None) and not self.is_activation and not self.is_transform:
            raise LayerInitializationException("Layer not initialised")

        self.initialised = True

    def _forward(self, input_data: LayerData, capture_data=False, batch_size=1) -> LayerData:
        raise NotImplementedError("Method not implemented")

    @required_init
    def forward(self, input_data: LayerData, capture_data=False, batch_size=1) -> LayerData:
        return self._forward(input_data, capture_data, batch_size=batch_size)

    def _backward(self, input_data: LayerData, output_data: LayerData,
                        previous_error: Gradients, batch_size=1, learning_rate=1) -> list[Gradients]:
        raise NotImplementedError("Method not implemented")


    @required_init
    def backward(self, input_data: LayerData, output_data: LayerData,
                       previous_error: Gradients, batch_size=1, learning_rate=1) -> list[Gradients]:
        return self._backward(input_data, output_data, previous_error, batch_size=batch_size, learning_rate=learning_rate)

    def _apply_gradients(self, weight_grads: np.ndarray, bias_grads: np.ndarray) -> None:
        raise NotImplementedError("Method not implemented")

    @required_init
    def apply_gradients(self, weight_grads: np.ndarray, bias_grads: np.ndarray) -> None:
        self._apply_gradients(weight_grads, bias_grads)