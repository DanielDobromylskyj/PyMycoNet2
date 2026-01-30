import numpy as np

from ..util.kernel import Kernel
from ..layers.default import DefaultLayer
from ..util.layer_data import LayerData, Gradients


class SingleStage(DefaultLayer):
    def __init__(self, function_name: str):
        super().__init__([
            Kernel(f"standard/acti/{function_name}.cl", targets=["forward"])
        ])

        self.function_name = function_name

        self.forward_kernel = self.get_kernel(f"standard/acti/{function_name}.cl")

    def _init(self):
        self.is_activation = True  # This layer is only for activations, we have no init data here. ( I hope )

    def _forward(self, input_data: LayerData, capture_data=False, batch_size=1) -> LayerData:
        outputs = input_data.duplicate_empty()

        self.forward_kernel.forward(
            self.queue, (batch_size, input_data.shape[0]), None,
            input_data.buffer.cl,
            outputs.buffer.cl,
            np.int32(input_data.shape[0]),
        )

        return outputs

    def _backward(self, input_data: LayerData, output_data: LayerData,
                        previous_error: Gradients, batch_size=1, learning_rate=1) -> list[Gradients]:
        raise NotImplementedError  # todo
