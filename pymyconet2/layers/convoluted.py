import numpy as np

from .default import DefaultLayer
from ..util.kernel import Kernel
from ..util.buffer import NetworkBuffer, EmptyNetworkBuffer
from ..activations import *
from ..util.layer_data import LayerData, Gradients


class Convoluted(DefaultLayer):
    def __init__(self, kernel_shape: tuple[int, int], stride: int, filter_count: int, activation: int | None = None, is_loading=False):
        """
            Activation value is just used for weight initialisation.
            No activation function is applied within this layer
        """
        super().__init__(
            kernels=[
                Kernel("standard/convoluted.cl", ["forward"]),
            ],
            is_loading=is_loading
        )

        self.forward_kernel = self.get_kernel("standard/convoluted.cl")
        #self.backward_kernel = self.get_kernel("training/convoluted.cl")

        self.kernel_shape = kernel_shape
        self.stride = stride
        self.filter_count = filter_count

        self.activation = activation

        self.weights_shape = (filter_count, *kernel_shape)
        self.bias_shape = (filter_count,)

    def __generate_weights(self) -> np.ndarray:
        count, fan_in, fan_out = self.weights_shape
        assert count == self.filter_count, "Weight shape is malformed"

        if self.activation == RELU:  # HE normal
            std = np.sqrt(2.0 / fan_in)
            weights = np.random.randn(count, fan_in, fan_out) * std

        elif self.activation in (SIGMOID, TANH, SOFTMAX, None):  # Xavier uniform
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            weights = np.random.uniform(
                -limit, limit, size=(count, fan_in, fan_out)
            )

        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

        return weights.astype(np.float32)

    def __generate_biases(self) -> np.ndarray:
        fan_out = self.bias_shape[0]

        if self.activation == RELU: # Often small positive bias helps dead ReLUs (I think)
            biases = np.full((fan_out,), 0.01, dtype=np.float32)
        else:
            biases = np.zeros((fan_out,), dtype=np.float32)

        return biases

    def _init(self):
        if not self.is_loading:
            weight_data = self.__generate_weights()
            self.weights = NetworkBuffer(self.ctx, self.queue, weight_data)

            bias_data = self.__generate_biases()
            self.biases = NetworkBuffer(self.ctx, self.queue, bias_data)

    def _forward(self, input_data: LayerData, capture_data=False, batch_size=1) -> LayerData:
        if len(input_data.shape) == 3:
            batch_count, input_width, input_height = input_data.shape
        else:
            input_width, input_height = input_data.shape

        output_width = input_width // self.stride
        output_height = input_height // self.stride
        total_outputs = output_width * output_height

        output = LayerData(self.ctx, self.queue, (batch_size, self.filter_count, output_width, output_height))

        event = self.forward_kernel.forward(
            self.queue, (batch_size, self.filter_count, total_outputs), None,

            input_data.buffer.cl,
            output.buffer.cl,
            self.weights.cl,
            self.biases.cl,

            np.int32(input_width),
            np.int32(input_height),

            np.int32(output_width),
            np.int32(output_height),

            np.int32(self.kernel_shape[0]),
            np.int32(self.kernel_shape[1]),

            np.int32(self.stride),
            np.int32(self.filter_count),
        )

        event.wait()



        return output

    def _backward(self, input_data: LayerData, output_data: LayerData,
                        previous_error: Gradients, batch_size=1, learning_rate=1) -> list[Gradients]:

        next_layer_error = Gradients(self.ctx, self.queue, (batch_size, self.input_size))
        next_layer_errors_unreduced = EmptyNetworkBuffer(
            self.ctx, self.queue,
            (batch_size, *self.weights_shape),
            item_dtype=np.float32
        )

        weight_gradiants = Gradients(self.ctx, self.queue, (batch_size, *self.weights_shape))
        bias_gradiants = Gradients(self.ctx, self.queue, (batch_size, *self.bias_shape))

        event = self.backward_kernel.backward(
            self.queue, (batch_size, *self.weights_shape), None,
            # Args
            input_data.buffer.cl,
            self.weights.cl,

            previous_error.gradiants.cl,
            next_layer_errors_unreduced.cl,
            weight_gradiants.gradiants.cl,
            bias_gradiants.gradiants.cl,

            np.int32(self.input_size),
            np.int32(self.output_size),
            np.float32(learning_rate)
        )

        self.backward_kernel.reducer(
            self.queue, (batch_size, self.input_size), None,
            # Args
            next_layer_errors_unreduced.cl,
            next_layer_error.gradiants.cl,

            np.int32(self.input_size),
            np.int32(self.output_size),

            wait_for=[event]
        )

        return [next_layer_error, weight_gradiants, bias_gradiants]

    def _apply_gradients(self, weight_grads: np.ndarray, bias_grads: np.ndarray) -> None:
        self.weights.write_to_buffer(self.weights.np - weight_grads)
        self.biases.write_to_buffer(self.biases.np - bias_grads)
