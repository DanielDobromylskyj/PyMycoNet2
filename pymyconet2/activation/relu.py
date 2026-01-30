from ..util.kernel import Kernel
from ..layers.default import DefaultLayer
from ..util.layer_data import LayerData, Gradients


class ReLU(DefaultLayer):
    def __init__(self):
        super().__init__([
            Kernel("standard/acti/relu.cl", targets=["relu"])
        ])

        self.forward_kernel = self.get_kernel("standard/acti/relu.cl")


    def _forward(self, input_data: LayerData, capture_data=False, batch_size=1) -> LayerData:
        outputs = input_data.duplicate_empty()

        self.forward_kernel.relu(
            self.queue, input_data.shape, None,
            input_data.buffer.cl,
            outputs.buffer.cl,
        )

        return outputs

    def _backward(self, input_data: LayerData, output_data: LayerData,
                        previous_error: Gradients, batch_size=1, learning_rate=1) -> list[Gradients]:
        raise NotImplementedError
