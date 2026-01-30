
__kernel void tanh(
    __global float* inputs,
    __global float* outputs
) {
    int i = get_global_index(0);
    outputs[i] = tanh(inputs[i]);
}
