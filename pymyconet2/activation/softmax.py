import numpy as np

from ..layers.default import DefaultLayer
from ..util.kernel import Kernel
from ..util.layer_data import LayerData, Gradients


class SoftMax(DefaultLayer):
    def __init__(self):
        super().__init__([
            Kernel(f"standard/acti/softmax.cl", targets=["forward"])
        ])

        self.forward_kernel = self.get_kernel(f"standard/acti/softmax.cl")

    def _init(self):
        self.is_activation = True

    def _forward(self, input_data: LayerData, capture_data=False, batch_size=1) -> LayerData:
        outputs = input_data.duplicate_empty()

        self.forward_kernel.forward(
            self.queue, (batch_size,), None,
            input_data.buffer.cl,
            outputs.buffer.cl,
            np.int32(input_data.shape[0])
        )

        return outputs
