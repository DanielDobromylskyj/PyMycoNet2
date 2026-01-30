
__kernel void relu(
    __global float* inputs,
    __global float* outputs
) {
    int i = get_global_index(0);

    float v = inputs[i];
    outputs[i] = (v <= 0) ? 0 : v;
}
