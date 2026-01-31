
// I cant put this function off any longer
// There is no way this indexing is correct first try

__kernel void backward(
    __global float* inputs,
    __global float* outputs,
    __global float* weights,

    __global float* previous_errors,
    __global float* next_errors,

    __global float* unreduced_weight_gradients,
    __global float* unreduced_bias_gradients,

    int input_width,
    int input_height,

    int output_width,
    int output_height,

    int kernel_width,
    int kernel_height,

    int stride,
    int filter_count,

    float learning_rate
) {
    int batch_index = get_global_id(0);
    int filter_index = get_global_id(1);
    int input_index = get_global_id(2);

    int input_y = input_index / input_width;
    int input_x = input_index % input_width;

    int output_x = input_x / stride;
    int output_y = input_y / stride;

    // Calculate Offsets
    int weight_offset = (kernel_width * kernel_height) * filter_index;
    int input_offset = (input_width * input_height) * batch_index;

    int output_filter_offset = (output_width * output_height) * filter_index;
    int output_batch_offset = (output_width * output_height * filter_count) * batch_index;
    int output_offset = output_batch_offset + output_filter_offset;

    int kernel_size = kernel_width * kernel_height;

    int weight_g_filter_offset = (kernel_size * (output_width * output_height)) * filter_index;
    int weight_g_batch_offset = kernel_size * (output_width * output_height) * filter_count;
    int weight_gradient_offset = weight_g_filter_offset + weight_g_batch_offset; // Is this right???

    float input_value = inputs[input_offset + input_index];

    // Start: output_x. Count: kernel_width / stride
    float next_error_sum = 0.0f;
    for (int dx=0; dx<(kernel_width / stride); dx++) {
        for (int dy=0; dy<(kernel_height / stride); dy++) {
            int output_index = (output_y + dy) * output_width + (output_x + dx);

            int kernel_x = input_x - (output_x * stride);
            int kernel_y = input_y - (output_y * stride);

            int weight_index = kernel_y * kernel_width + kernel_x;
            int weight_grad_index = (kernel_size * output_index) + weight_index;

            float previous_error = previous_errors[output_offset + output_index];
            float weight = weights[weight_offset + weight_index];

            float weight_gradient = input_value * previous_error * learning_rate;
            unreduced_weight_gradients[weight_gradient_offset + weight_index] = weight_gradient;
        }
    }
    
}
