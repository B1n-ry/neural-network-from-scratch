
extern "C" __global__ void img_result(const unsigned char* images, const char* labels, const float* network, float* costs, const size_t num_el) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_el) return;

    /* costs[i] = network[i]; */

    const unsigned char* image = images + (i * 784);
    const unsigned char label = labels[i];

    float* layer1_values = new float[16];
    float* layer2_values = new float[16];
    float* layer3_values = new float[10];

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

    delete[] layer1_values;
    delete[] layer2_values;
    delete[] layer3_values;
}