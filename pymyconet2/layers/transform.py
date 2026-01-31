from .default import DefaultLayer
from ..util.layer_data import LayerData


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



