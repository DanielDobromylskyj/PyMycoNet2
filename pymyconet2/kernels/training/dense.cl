
__kernel void backward(
    __global float* inputs,
    __global float* weights,

    __global float* previous_errors,
    __global float* next_errors,
    __global float* weight_gradients,
    __global float* bias_gradients,

    int input_size,
    int output_size,
    float learning_rate
) {
    int batch_index = get_global_id(0);
    int input_index = get_global_id(1);
    int output_index = get_global_id(2);

    int weight_index = input_index * output_size + output_index;

    int input_offset = input_size * batch_index;
    int output_offset = output_size * batch_index;

    int weight_gradient_offset = input_size * output_size * batch_index;
    int previous_error_gradients_offset = output_size * batch_index;
    int bias_gradient_offset = output_size * batch_index;

    float weight = weights[weight_index];

    float delta = previous_errors[previous_error_gradients_offset + output_index];
    float weight_gradient = delta * inputs[input_offset + input_index] * learning_rate;

    weight_gradients[weight_gradient_offset + weight_index] = weight_gradient;

    // Yes, we use weight index, its reduced later
    next_errors[weight_gradient_offset + weight_index] = weight * delta;

    if (input_index == 0) { // Only run this once per output node
        bias_gradients[bias_gradient_offset + output_index] = delta * learning_rate;
    }
}

__kernel void reducer(
    __global float* unreduced_error_gradients,
    __global float* reduced_error_gradients,
    int input_size,
    int output_size
) {
    int batch_index = get_global_id(0);
    int input_index = get_global_id(1);

    int unreduced_gradient_offset = input_size * output_size * batch_index;
    int reduced_gradient_offset = input_size * batch_index;

    float total = 0.0f;
    for (int output_index=0; output_index<output_size; output_index++) {
        int weight_index = input_index * output_size + output_index;
        total += unreduced_error_gradients[unreduced_gradient_offset + weight_index];
    }

    reduced_error_gradients[reduced_gradient_offset + input_index] = total;
}