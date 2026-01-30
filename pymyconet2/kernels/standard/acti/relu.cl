
__kernel void forward(
    __global float* inputs,
    __global float* outputs,
    int batch_size
) {
    int batch_index = get_global_id(0);
    int i = get_global_id(1);
    int offset = batch_index * batch_size;

    float v = inputs[offset + i];
    outputs[offset + i] = (v <= 0) ? 0 : v;
}

__kernel void backward(
    __global float* activated,
    __global float* error_out,
    __global float* error_in,
    int batch_size
) {
    int batch_index = get_global_id(0);
    int i = get_global_id(1);
    int offset = batch_index * batch_size;

    float derivative = activated[offset + i] > 0 ? 1.0f : 0.0f;
    error_out[offset + i] = derivative * error_in[offset + i];
}
