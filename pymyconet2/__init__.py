import numpy as np
import pyopencl as cl

from .util.layer_data import LayerData, Gradients
from .layers.default import DefaultLayer
from .util.kernel import Kernel, KernelBuilder
from .activations import *

class NetworkValidationException(Exception):
    pass

class NetworkInitializationException(Exception):
    pass


class Network:
    def __init__(self, layout, validate=True, context=None, queue=None):
        if not context or not queue:
            context = cl.create_some_context()
            queue = cl.CommandQueue(context)

        self.context = context
        self.queue = queue

        self.layout = layout

        #self.gradient_summing = Kernel("util/network.cl", targets=[])

        # Flags
        self.flag_last_layer_is_softmax = False

        # Validation - Check to see if the network has some problems
        if validate:
            self.validate()

        self.initialise()

    def initialise(self):
        print("[Network] Initialising...")
        kernel_builder = KernelBuilder(self.context)

        for layer in self.layout:
            for kernel_to_load in layer.kernels:
                if not isinstance(kernel_to_load, Kernel):
                    raise NetworkInitializationException(f"Couldn't initialise kernel, Invalid data: {kernel_to_load}")

                kernel_builder.build_kernel(kernel_to_load)
            layer.init(self.context, self.queue)



        last_layer = self.layout[-1]
        if hasattr(last_layer, "activation"):
            if last_layer.activation == SOFTMAX:
                self.flag_last_layer_is_softmax = True

    def validate(self):
        for i, layer in enumerate(self.layout):
            if not isinstance(layer, DefaultLayer):
                raise NetworkValidationException(f"Layer {i+1} is not recognized as a layer.")

            if hasattr(layer, "activation"):
                if layer.activation == SOFTMAX and (len(self.layout) - 1 != i):
                    raise NetworkValidationException(f"Layer {i+1} is using Softmax but is not an output layer.")

    def _forward_no_capture(self, next_layer_inputs: LayerData, batch_size: int) -> LayerData:
        for i, layer in enumerate(self.layout):
            next_layer_inputs = layer.forward(next_layer_inputs, capture_data=False, batch_size=batch_size)
        return next_layer_inputs

    def _forward_with_capture(self, next_layer_inputs: LayerData, batch_size: int) -> tuple[LayerData, list[LayerData]]:
        layer_data = [next_layer_inputs]

        for i, layer in enumerate(self.layout):
            next_layer_inputs = layer.forward(next_layer_inputs, capture_data=True, batch_size=batch_size)
            layer_data.append(next_layer_inputs)

        return next_layer_inputs, layer_data

    def forward_single(self, inputs: list | np.ndarray) -> np.ndarray:
        if type(inputs) is list:
            inputs = np.array(inputs, dtype=np.float32)

        input_data = LayerData(self.context, self.queue, inputs.shape, inputs)

        output_data = self._forward_no_capture(input_data, batch_size=1)

        return output_data.buffer.np[0]

    def forward_many(self, inputs: list[np.ndarray] | np.ndarray, batch_size=None) -> np.ndarray:
        if type(inputs) is list:
            inputs = np.array(inputs, dtype=np.float32)

        if not batch_size:
            batch_size = inputs.shape[0]

        input_data = LayerData(self.context, self.queue, inputs.shape, inputs)
        output_data = self._forward_no_capture(input_data, batch_size=batch_size)
        return output_data.buffer.np

    def _backward(self):
        pass

    @staticmethod
    def average_batched(batched_values: np.ndarray) -> np.ndarray:
        return batched_values.mean(axis=0)

    def average_and_convert_gradients_to_numpy(self, layer_gradients):
        return [
            (
                self.average_batched(layer[0].gradiants.np),
                self.average_batched(layer[1].gradiants.np),
            ) for layer in layer_gradients
        ]

    def train_batched_epoche(self,
                             inputs_batch: list[np.ndarray] | np.ndarray,
                             targets_batch: list[np.ndarray] | np.ndarray,
                             learning_rate: float) -> list[tuple[Gradients]]:

        if type(inputs_batch) is list:
            inputs_batch = np.array(inputs_batch, dtype=np.float32)

        if type(targets_batch) is list:
            targets_batch = np.array(targets_batch, dtype=np.float32)

        batch_size = inputs_batch.shape[0]

        input_data = LayerData(self.context, self.queue, inputs_batch.shape, inputs_batch)
        outputs_batch, other_data = self._forward_with_capture(input_data, batch_size)

        # I have a flag self.flag_last_layer_is_softmax that can be used here
        errors = outputs_batch.buffer.np - targets_batch

        next_errors = Gradients(self.context, self.queue, errors.shape)
        next_errors.write(errors.astype(np.float32))


        gradients = []

        for layer_index in range(len(self.layout)-1, -1, -1):
            layer = self.layout[layer_index]

            layer_input = other_data[layer_index]
            layer_output = other_data[layer_index+1]

            next_errors, weight_gradiants, bias_gradiants = layer.backward(
                layer_input,
                layer_output,
                next_errors,
                batch_size=batch_size,
                learning_rate=learning_rate
            )

            gradients.insert(0, (weight_gradiants, bias_gradiants))

        return gradients

    def apply_gradients(self, gradients):
        for i, layer in enumerate(self.layout):
            layer.apply_gradients(*gradients[i])

    def train(self, train_inputs: list[np.ndarray] | np.ndarray, train_targets: list[np.ndarray] | np.ndarray, epoches: int, max_batch_size: int, learning_rate: float):
        if len(train_inputs) > max_batch_size:
            train_inputs = [ train_inputs[i:i + max_batch_size] for i in range(0, len(train_inputs), max_batch_size) ]
            train_targets = [ train_targets[i:i + max_batch_size] for i in range(0, len(train_targets), max_batch_size) ]
        else:
            train_inputs = [train_inputs]
            train_targets = [train_targets]

        for epoch in range(epoches):
            #print(f"\r[Network] Training epoche {epoch+1}/{epoches}", end="")

            all_batch_gradients = [
                self.average_and_convert_gradients_to_numpy(
                    self.train_batched_epoche(
                        train_inputs[batch_index],
                        train_targets[batch_index],
                        learning_rate
                    )
                ) for batch_index in range(len(train_inputs))
            ]

            for gradient_batch in all_batch_gradients:
                self.apply_gradients(gradient_batch)

        #print()

