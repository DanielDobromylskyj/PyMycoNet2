from ..util.kernel import Kernel
from ..layers.default import DefaultLayer
from ..util.layer_data import LayerData, Gradients


class SingleStage(DefaultLayer):
    def __init__(self, function_name: str):
        super().__init__([
            Kernel(f"standard/acti/{function_name}.cl", targets=[function_name])
        ])

        self.function_name = function_name
        self.forward_kernel = self.get_kernel(f"standard/acti/{function_name}.cl")


    def _forward(self, input_data: LayerData, capture_data=False, batch_size=1) -> LayerData:
        outputs = input_data.duplicate_empty()

        self.forward_kernel.__getattr__(self.function_name)(
            self.queue, input_data.shape, None,
            input_data.buffer.cl,
            outputs.buffer.cl,
        )

        return outputs

    def _backward(self, input_data: LayerData, output_data: LayerData,
                        previous_error: Gradients, batch_size=1, learning_rate=1) -> list[Gradients]:
        raise NotImplementedError  # todo
