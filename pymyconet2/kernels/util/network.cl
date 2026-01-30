
__kernel void total_across_batch(
    __global float* input_data,
    __global float* output_data,

    int offset,
    int skip_size,
    int skip_quantity
) {
    float total = 0.0;

    for (int i=0; i<skip_quantity; i++) {
        total += input_data[offset + (skip_size * i)];
    }

    output_data[offset] = total;
}
