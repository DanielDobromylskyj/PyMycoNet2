

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
