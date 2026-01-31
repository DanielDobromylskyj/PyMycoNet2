from .default import DefaultLayer
from ..util.layer_data import LayerData, Gradients


class Condense(DefaultLayer):
    def __init__(self):
        super().__init__( kernels=[] )

        self.is_transform = True

    def _init(self):
        pass

    def _forward(self, input_data: LayerData, capture_data=False, batch_size=1) -> LayerData:
        batch_count = input_data.shape[0]

        # Reversing this function:
        # original_shape = input_data.shape
        # restored = flattened_data.reshape(original_shape)

        flattened = input_data.buffer.np.reshape(batch_count, -1)
        return LayerData(self.ctx, self.queue, flattened.shape, data=flattened)

    def _backward(self, input_data: LayerData, output_data: LayerData,
                        previous_error: Gradients, batch_size=1, learning_rate=1) -> list[Gradients]:

        original_shape = input_data.shape

        try:
            restored = previous_error.gradiants.np.reshape(original_shape)

        except ValueError:
            raise ValueError(f"Failed to un-condense gradients. Shape: {original_shape}\nPlease ensure the layers after your convolutions have the correct input sizes.")

        grads = Gradients.from_layer_data(LayerData(
            self.ctx, self.queue, restored.shape, data=restored)
        )

        return [grads, None, None]
