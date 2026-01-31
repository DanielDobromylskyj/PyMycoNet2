
__kernel void forward(
    __global float* inputs,
    __global float* outputs,
    int output_size
) {
    int batch_index = get_global_id(0);
    int base = batch_index * output_size;

    // SoftMax -> activated_i = exp(x_i - max_x) / sum_j exp(x_j - max_x)

    // Find max value
    float max_val = inputs[base];
    for (int i = 1; i < output_size; i++) {
        float v = inputs[base + i];
        if (v > max_val)
            max_val = v;
    }

    // Exp & Sum
    float sum = 0.0f;
    for (int i = 0; i < output_size; i++) {
        float e = exp(inputs[base + i] - max_val);
        outputs[base + i] = e;
        sum += e;
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < output_size; i++) {
        outputs[base + i] *= inv_sum;
    }
}