import numpy as np

from .buffer import NetworkBuffer, EmptyNetworkBuffer


class LayerData:
    def __init__(self, ctx, queue, shape, data: np.ndarray | None=None):
        if data is None:
            self.buffer = EmptyNetworkBuffer(ctx, queue, shape, item_dtype=np.float32)
        else:
            self.buffer = NetworkBuffer(ctx, queue, data)

        self.extra_data = []

    def add_extra_data(self, layer_data) -> None:
        self.extra_data.append(layer_data.buffer)

    def duplicate_empty(self):
        return LayerData(self.buffer.cl_ctx, self.buffer.cl_queue, self.buffer.shape)

    @property
    def shape(self):
        return self.buffer.shape


class Gradients:
    def __init__(self, ctx, queue, shape):
        self.gradiants = EmptyNetworkBuffer(ctx, queue, shape, item_dtype=np.float32)

    def write(self, numpy_buffer):
        self.gradiants.write_to_buffer(numpy_buffer, offset=0)
