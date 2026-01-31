
// I hate writing kernels.
// Why do I do this to myself?

__kernel void forward(
    __global float* inputs,
    __global float* outputs,

    __global float* weights,
    __global float* biases,

    int input_width,
    int input_height,

    int output_width,
    int output_height,

    int kernel_width,
    int kernel_height,

    int stride,
    int filter_count

) {
    int batch_index = get_global_id(0); // Batching makes this so much more complicated. (For the performance!)
    int filter_index = get_global_id(1);
    int output_index = get_global_id(2);

    // Calculate X/Y Coords
    int output_y = output_index / output_width;
    int output_x = output_index % output_width;

    int input_x = output_x * stride;
    int input_y = output_y * stride;

    // Calculate Offsets
    int weight_offset = (kernel_width * kernel_height) * filter_index;

    int input_offset = (input_width * input_height) * batch_index;

    int output_filter_offset = (output_width * output_height) * filter_index;
    int output_batch_offset = (output_width * output_height * filter_count) * batch_index;
    int output_offset = output_batch_offset + output_filter_offset;


    // Calculate weighted sum of output node
    float weighted_sum = biases[filter_index];
    for (int dx=0; dx<kernel_width; dx++) {
        for (int dy=0; dy<kernel_height; dy++) {
            int input_index = (input_y + dy) * input_width + (input_x + dx);
            int weight_index = dy * kernel_width + dx;

            float weighted = inputs[input_offset + input_index] * weights[weight_offset + weight_index];
            weighted_sum += weighted;
        }
    }

    outputs[output_offset + output_index] = weighted_sum;
}
