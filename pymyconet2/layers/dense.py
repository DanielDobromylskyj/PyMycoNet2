import numpy as np

from .default import DefaultLayer
from ..util.kernel import Kernel
from ..util.buffer import NetworkBuffer, EmptyNetworkBuffer
from ..activations import *
from ..util.layer_data import LayerData, Gradients


class DenseLayer(DefaultLayer):
    def __init__(self, input_size: int, output_size: int, activation: int, is_loading=False):
        super().__init__(
            kernels=[
                Kernel("standard/dense.cl", ["forward", "reduce", "softmax_activation"]),
                Kernel("training/dense.cl", ["backward", "reducer"])
            ],
            is_loading=is_loading
        )

        self.forward_kernel = self.get_kernel("standard/dense.cl")
        self.backward_kernel = self.get_kernel("training/dense.cl")

        self.input_size = input_size
        self.output_size = output_size

        self.weights_shape = (input_size, output_size)
        self.bias_shape = (output_size,)

        self.activation = activation

    def __generate_weights(self) -> np.ndarray:
        fan_in, fan_out = self.weights_shape

        if self.activation == RELU:  # HE Normal
            std = np.sqrt(2.0 / fan_in)
            weights = np.random.randn(fan_in, fan_out) * std

        elif self.activation in (SIGMOID, TANH, SOFTMAX): # Xavier / Glorot uniform
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            weights = np.random.uniform(-limit, limit, size=(fan_in, fan_out))

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
        output = LayerData(self.ctx, self.queue, (batch_size, self.output_size))
        output_unactivated = LayerData(self.ctx, self.queue, (batch_size, self.output_size))
        output_unreduced = LayerData(self.ctx, self.queue, (batch_size, *self.weights_shape))

        event = self.forward_kernel.forward(
            self.queue, (batch_size, *self.weights_shape), None,

            # Args
            input_data.buffer.cl,
            output_unreduced.buffer.cl,
            self.weights.cl,  # Not LayerData, so no .buffer to find, straight to the source

            np.int32(self.input_size),
            np.int32(self.output_size)
        )

        event2 = self.forward_kernel.reduce(
            self.queue, (batch_size, self.output_size), None,
            # Args
            output_unreduced.buffer.cl,
            output_unactivated.buffer.cl,
            output.buffer.cl,
            self.biases.cl,

            np.int32(self.input_size),
            np.int32(self.output_size),
            np.int32(self.activation),

            wait_for=[event]
        )

        if self.activation == SOFTMAX:  # Look at me im special
            self.forward_kernel.softmax_activation(
                self.queue, (batch_size,), None,

                # Args
                output.buffer.cl,
                np.int32(self.output_size),

                wait_for=[event2]
            ).wait()

        else:
            event2.wait()

        if capture_data:
            output.add_extra_data(output_unactivated)

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

        outputs_unactivated = output_data.extra_data[0]

        event = self.backward_kernel.backward(
            self.queue, (batch_size, *self.weights_shape), None,
            # Args
            input_data.buffer.cl,
            output_data.buffer.cl,
            outputs_unactivated.cl,
            self.weights.cl,

            previous_error.gradiants.cl,
            next_layer_errors_unreduced.cl,
            weight_gradiants.gradiants.cl,
            bias_gradiants.gradiants.cl,

            np.int32(self.input_size),
            np.int32(self.output_size),
            np.int32(self.activation),
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
