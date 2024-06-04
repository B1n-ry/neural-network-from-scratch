/**
 * img_result.cu
 * This file contains the CUDA kernel that calculates the result of the neural network for each image.
 * This code is made to only work with neural networks with 784 input nodes, 16 nodes in the first hidden layer,
 * 16 nodes in the second hidden layer, and 10 output nodes.
 * So the network contains 1 input layer, 2 hidden layers, and 1 output layer.
*/

__device__ float get_weight(const float* network, const unsigned int layer, const unsigned int i1, const unsigned int i2);

__device__ float sigmoid(float x);
__device__ float d_sigmoid(float x);

__device__ float d_w_l3(const float* l3_values, const float* l2_values, const float* expected, const unsigned int i3, const unsigned int i2);
__device__ float d_b_l3(const float* l3_values, const float* expected, const unsigned int i3);

__device__ float d_v_l2(const float* network, const float* l3_values, const float* expected, const unsigned int i2);
__device__ float d_w_l2(const float* network, const float* l3_values, const float* l2_values, const float* l1_values, const float* expected,
        const unsigned int i2, const unsigned int i1);
__device__ float d_b_l2(const float* network, const float* l3_values, const float* l2_values, const float* expected, const unsigned int i2);

__device__ float d_v_l1(const float* network, const float* l3_values, const float* l2_values, const float* expected, const unsigned int i1);
__device__ float d_w_l1(const float* network, const float* l3_values, const float* l2_values, const float* l1_values,
        const unsigned char* l0_values, const float* expected, const unsigned int i1, const unsigned int i0);
__device__ float d_b_l1(const float* network, const float* l3_values, const float* l2_values, const float* l1_values,
        const float* expected, const unsigned int i1);


extern "C" __global__ void img_result(const unsigned char* images, const char* labels, const float* network, float* nodes, float* costs, const size_t num_el) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_el) return;

//    float* network_alterations = derivatives + (i * 13002);  // 13002 is the number of all weights and biases in the network

    const unsigned char* image = images + (i * 784);
    const unsigned char label = labels[i];

    float expected[10] = {0};
    expected[label] = 1;

    float* layer1_values = nodes + (i * (16 + 16 + 10));
    float* layer2_values = nodes + (i * (16 + 16 + 10) + 16);
    float* layer3_values = nodes + (i * (16 + 16 + 10) + 16 + 16);

    unsigned int offset = 0;
    for (int n = 0; n < 16; n++) {
        float sum = network[n * 785 + offset];  // bias
        for (int j = 0; j < 784; j++) {
            sum += network[1 + (n * 785) + j + offset] * (image[j] / 255.0f);  // weights
        }
        layer1_values[n] = sigmoid(sum);
    }
    offset += 785 * 16;
    for (int n = 0; n < 16; n++) {
        float sum = network[n * 17 + offset];  // bias
        for (int j = 0; j < 16; j++) {
            sum += network[1 + (n * 17) + j + offset] * layer1_values[j];  // weights
        }
        layer2_values[n] = sigmoid(sum);
    }
    offset += 17 * 16;
    for (int n = 0; n < 10; n++) {
        float sum = network[n * 17 + offset];  // bias
        for (int j = 0; j < 16; j++) {
            sum += network[1 + (n * 17) + j + offset] * layer2_values[j];  // weights
        }
        layer3_values[n] = sigmoid(sum);
    }

    costs[i] = 0.0;
    for (unsigned int n = 0; n < 10; n++) {
        costs[i] += pow(expected[n] - layer3_values[n], 2);
    }
}

extern "C" __global__ void backpropagation(const float* network, const float* nodes, const unsigned char* images, const char* labels, float* derivatives, const size_t num_el) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_el) return;

    const unsigned char* image = images + (i * 784);
    const unsigned char label = labels[i];

    float expected[10] = {0};
    expected[label] = 1;

    const float* layer1_values = nodes + (i * (16 + 16 + 10));
    const float* layer2_values = nodes + (i * (16 + 16 + 10) + 16);
    const float* layer3_values = nodes + (i * (16 + 16 + 10) + 16 + 16);

    float* network_alterations = derivatives + (i * 13002);  // 13002 is the number of all weights and biases in the network

    // Calculate derivatives/changes
    unsigned int offset = 0;
    for (int n = 0; n < 16; n++) {
        network_alterations[n * 785 + offset] = d_b_l1(network, layer3_values, layer2_values, layer1_values, expected, n);  // bias
        for (int j = 0; j < 784; j++) {
            network_alterations[1 + (n * 785) + j + offset] = d_w_l1(network, layer3_values, layer2_values, layer1_values, image, expected, n, j);  // weights
        }
    }
    offset += 785 * 16;
    for (int n = 0; n < 16; n++) {
        network_alterations[n * 17 + offset] = d_b_l2(network, layer3_values, layer2_values, expected, n);  // bias
        for (int j = 0; j < 16; j++) {
            network_alterations[1 + (n * 17) + j + offset] = d_w_l2(network, layer3_values, layer2_values, layer1_values, expected, n, j);  // weights
        }
    }
    offset += 17 * 16;
    for (int n = 0; n < 10; n++) {
        network_alterations[n * 17 + offset] = d_b_l3(layer3_values, expected, n);  // bias
        for (int j = 0; j < 16; j++) {
            network_alterations[1 + (n * 17) + j + offset] = d_w_l3(layer3_values, layer2_values, expected, n, j);  // weights
        }
    }
}

extern "C" __global__ void average_gradient(const float* derivatives, float* avg_derivatives, const size_t dataset_size, const size_t num_el) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_el) return;

    float sum = 0;
    for (unsigned int j = 0; j < dataset_size; j++) {
        sum += derivatives[j * num_el + i];
    }
    avg_derivatives[i] = sum / dataset_size;
}

__device__ float get_weight(const float* network, const unsigned int layer, const unsigned int i1, const unsigned int i2) {
    switch (layer) {
        case 1:
            return network[i2 * 785 + i1 + 1];
        case 2:
            return network[i2 * 17 + i1 + 1 + 785 * 16];
        case 3:
            return network[i2 * 17 + i1 + 1 + 785 * 16 + 17 * 16];
        default:
            return 0;
    }
}

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}
__device__ float d_sigmoid(float sigmoided) {
    return sigmoided * (1.0f - sigmoided);  // Since d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
}

__device__ float d_w_l3(const float* l3_values, const float* l2_values, const float* expected, const unsigned int i3, const unsigned int i2) {
    float sigmoided = l3_values[i3];
    return l2_values[i2] * d_sigmoid(sigmoided) * 2 * (l3_values[i3] - expected[i3]);
}
__device__ float d_b_l3(const float* l3_values, const float* expected, const unsigned int i3) {
    float sigmoided = l3_values[i3];
    return d_sigmoid(sigmoided) * 2 * (l3_values[i3] - expected[i3]);
}
__device__ float d_v_l2(const float* network, const float* l3_values, const float* expected, const unsigned int i2) {
    float sum = 0;
    for (unsigned int i3 = 0; i3 < 10; i3++) {
        sum += get_weight(network, 3, i2, i3) * d_sigmoid(l3_values[i3]) * 2 * (l3_values[i3] - expected[i3]);
    }
    return sum;
}
__device__ float d_w_l2(const float* network, const float* l3_values, const float* l2_values, const float* l1_values, const float* expected,
        const unsigned int i2, const unsigned int i1) {
    float sigmoided = l2_values[i2];
    return l1_values[i1] * d_sigmoid(sigmoided) * d_v_l2(network, l3_values, expected, i2);
}
__device__ float d_b_l2(const float* network, const float* l3_values, const float* l2_values, const float* expected, const unsigned int i2) {
    float sigmoided = l2_values[i2];
    return d_sigmoid(sigmoided) * d_v_l2(network, l3_values, expected, i2);
}
__device__ float d_v_l1(const float* network, const float* l3_values, const float* l2_values, const float* expected, const unsigned int i1) {
    float sum = 0;
    for (unsigned int i2 = 0; i2 < 16; i2++) {
        sum += get_weight(network, 2, i1, i2) * d_sigmoid(l2_values[i2]) * d_v_l2(network, l3_values, expected, i2);
    }
    return sum;
}
__device__ float d_w_l1(const float* network, const float* l3_values, const float* l2_values, const float* l1_values,
        const unsigned char* l0_values, const float* expected, const unsigned int i1, const unsigned int i0) {
    float sigmoided = l1_values[i1];
    return (l0_values[i0] / 255.0f) * d_sigmoid(sigmoided) * d_v_l1(network, l3_values, l2_values, expected, i1);
}
__device__ float d_b_l1(const float* network, const float* l3_values, const float* l2_values, const float* l1_values,
        const float* expected, const unsigned int i1) {
    float sigmoided = l1_values[i1];
    return d_sigmoid(sigmoided) * d_v_l1(network, l3_values, l2_values, expected, i1);
}
