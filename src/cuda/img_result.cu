
extern "C" __global__ void img_result(const unsigned char* images, const char* labels, float* nodes, float* networks, float* costs, const size_t num_el) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_el) return;

    float* network = networks + (i * (785 * 16 + 17 * 16 + 17 * 10));

    const unsigned char* image = images + (i * 784);
    const unsigned char label = labels[i];

    float* layer1_values = nodes + (i * (16 + 16 + 10));
    float* layer2_values = nodes + (i * (16 + 16 + 10) + 16);
    float* layer3_values = nodes + (i * (16 + 16 + 10) + 16 + 16);

    unsigned int offset = 0;

    for (int n = 0; n < 16; n++) {
        float sum = network[n * 785 + offset];
        for (int j = 0; j < 784; j++) {
            sum += network[1 + (n * 785) + j + offset] * image[j];
        }
        layer1_values[n] = 1.0f / (1.0f + exp(-sum));
    }
    offset += 785 * 16;
    for (int n = 0; n < 16; n++) {
        float sum = network[n * 17 + offset];
        for (int j = 0; j < 16; j++) {
            sum += network[1 + (n * 17) + j + offset] * layer1_values[j];
        }
        layer2_values[n] = 1.0f / (1.0f + exp(-sum));
    }
    offset += 17 * 16;
    for (int n = 0; n < 10; n++) {
        float sum = network[n * 17 + offset];
        for (int j = 0; j < 16; j++) {
            sum += network[1 + (n * 17) + j + offset] * layer2_values[j];
        }
        layer3_values[n] = 1.0f / (1.0f + exp(-sum));
    }

    costs[i] = 0.0;
    for (unsigned int n = 0; n < 10; n++) {
        costs[i] += pow((float) (n == label) - layer3_values[n], 2);
    }
}

extern "C" __global__ void copy_network(const float* network, float* network_copy, const size_t network_size, const size_t num_el) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_el) return;

    for (size_t j = 0; j < network_size; j++) {
        network_copy[i * network_size + j] = network[j];
    }
}