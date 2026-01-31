

__kernel void forward(
    __global float* inputs,
    __global float* outputs_unreduced,
    __global float* weights,

    int input_size,
    int output_size
) {
    int batch_index = get_global_id(0);
    int input_index = get_global_id(1);
    int output_index = get_global_id(2);

    int input_offset = batch_index * input_size;
    int output_offset = batch_index * (input_size * output_size);

    int weight_index = (output_index * input_size) + input_index;

    float weighted = inputs[input_offset + input_index] * weights[weight_index];
    outputs_unreduced[output_offset + weight_index] = weighted;
}


__kernel void reduce(
    __global float* outputs_unreduced,
    __global float* outputs_unactivated,
    __global float* biases,

    int input_size,
    int output_size
) {
    int batch_index = get_global_id(0);
    int output_index = get_global_id(1);

    int reduced_offset = batch_index * output_size;
    int unreduced_offset = batch_index * (input_size * output_size);

    // Sum and Bias
    float total = biases[output_index];
    for (int input_index = 0; input_index<input_size; input_index++) {
        int unreduced_index = (output_index * input_size) + input_index;
        total = total + outputs_unreduced[unreduced_offset + unreduced_index];
    }

    outputs_unactivated[reduced_offset + output_index] = total;
    //outputs_reduced[reduced_offset + output_index] = activated;
}


// Due to SoftMax being a bitch, We have to run another kernel pass over ALL the data
// For this reason, its not recommended to use softmax on large layers, and only recommended for small outputs
// This is basically being run on 1 thread per batch, Which is shit for performance. (Idk how to do it better)
// This things only saving grace is that in runs multiple batches in parallel.
__kernel void softmax_activation(
    __global float* outputs,
    int output_size
) {
    int batch_index = get_global_id(0);
    int base = batch_index * output_size;

    // SoftMax -> activated_i = exp(x_i - max_x) / sum_j exp(x_j - max_x)

    // Find max value
    float max_val = outputs[base];
    for (int i = 1; i < output_size; i++) {
        float v = outputs[base + i];
        if (v > max_val)
            max_val = v;
    }

    // Exp & Sum
    float sum = 0.0f;
    for (int i = 0; i < output_size; i++) {
        float e = exp(outputs[base + i] - max_val);
        outputs[base + i] = e;
        sum += e;
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < output_size; i++) {
        outputs[base + i] *= inv_sum;
    }
}
