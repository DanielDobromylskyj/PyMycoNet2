import pyopencl as cl
import numpy

mf = cl.mem_flags

def multiply_array(shape):
    if type(shape) == int:
        return shape

    total = 1

    for x in shape:
        total *= x

    return total

class EmptyNetworkBuffer:
    def __init__(self, ctx, queue, shape, item_dtype):
        self.cl_ctx, self.cl_queue = ctx, queue
        self.__shape = shape
        self.__dtype = item_dtype
        self.__item_count = multiply_array(shape)
        self.__size = self.__item_count * numpy.dtype(item_dtype).itemsize

        self.__cl_buffer = cl.Buffer(self.cl_ctx, mf.READ_WRITE, size=self.__size, hostbuf=None)
        self.__np_buffer = numpy.empty(self.__shape, dtype=self.__dtype)

        self.__last_sync = None
        self.__sync_np_to_cl()


    @property
    def cl(self):
        if self.__last_sync != "cl":
            self.__sync_np_to_cl()
        return self.__cl_buffer

    @property
    def np(self):
        if self.__last_sync != "np":
            self.__sync_cl_to_np()
        return self.__np_buffer


    @property
    def dtype(self):
        return self.__dtype

    @property
    def size(self):
        return self.__size

    @property
    def shape(self):
        return self.__shape

    @property
    def sync_location(self):
        return self.__last_sync


    def __sync_np_to_cl(self):  # todo - see about making a function that returns the enqueue event (for async stuff)
        cl.enqueue_copy(self.cl_queue, self.__cl_buffer, self.__np_buffer).wait()
        self.__last_sync = "cl"

    def __sync_cl_to_np(self):  # todo - see about making a function that returns the enqueue event (for async stuff)
        cl.enqueue_copy(self.cl_queue, self.__np_buffer, self.__cl_buffer).wait()
        self.__last_sync = "np"


    def write_to_buffer(self, array: numpy.ndarray, offset=0):
        if self.__last_sync == "np":
            end = offset + array.size
            self.__np_buffer[offset:end] = array.astype(self.__dtype, copy=False)

        if self.__last_sync == "cl":
            cl.enqueue_copy(self.cl_queue, self.cl, array, device_offset=offset * numpy.dtype(self.__dtype).itemsize)


class NetworkBuffer(EmptyNetworkBuffer):
    def __init__(self, ctx, queue, array: numpy.ndarray):
        super().__init__(ctx, queue, array.shape, array.dtype)
        self.write_to_buffer(array, offset=0)