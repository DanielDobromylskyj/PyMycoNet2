

__kernel void forward(
    __global float* inputs,
    __global float* outputs,
    int batch_size
) {
    int batch_index = get_global_id(0);
    int i = get_global_id(1);
    int offset = batch_index * batch_size;

    outputs[offset + i] = 1.0f / (1.0f + exp(-inputs[offset + i]));
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

    float activated_output = activated[offset + i];
    float derivative = activated_output * (1.0f - activated_output);
    error_out[offset + i] = derivative * error_in[offset + i];
}
