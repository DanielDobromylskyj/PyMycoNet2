import numpy as np

from ..util.kernel import Kernel
from ..layers.default import DefaultLayer
from ..util.layer_data import LayerData, Gradients


class SingleStage(DefaultLayer):
    def __init__(self, function_name: str):
        super().__init__([
            Kernel(f"standard/acti/{function_name}.cl", targets=["forward", "backward"])
        ])

        self.function_name = function_name

        self.general_kernel = self.get_kernel(f"standard/acti/{function_name}.cl")

    def _init(self):
        self.is_activation = True  # This layer is only for activations, we have no init data here. ( I hope )

    def _apply_gradients(self, weight_grads: np.ndarray, bias_grads: np.ndarray) -> None:
        pass

    def _forward(self, input_data: LayerData, capture_data=False, batch_size=1) -> LayerData:
        outputs = input_data.duplicate_empty()
        target_shape = input_data.shape[-1]

        self.general_kernel.forward(
            self.queue, (batch_size, target_shape), None,
            input_data.buffer.cl,
            outputs.buffer.cl,
            np.int32(target_shape),
        )

        return outputs

    def _backward(self, input_data: LayerData, output_data: LayerData,
                        previous_error: Gradients, batch_size=1, learning_rate=1) -> list[Gradients]:
        next_errors = previous_error.duplicate_empty()
        target_shape = previous_error.gradiants.shape[-1]

        self.general_kernel.backward(
            self.queue, (batch_size, target_shape), None,
            output_data.buffer.cl,
            next_errors.gradiants.cl,
            previous_error.gradiants.cl,
            np.int32(target_shape)
        )

        return [next_errors, None, None]
