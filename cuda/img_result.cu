#include <stdlib.h>

template<int N>
struct NeuralNode {
    float bias;
    float weights[N];
};


struct NeuralNetwork {
    struct NeuralNode<784> layers1[16];
    struct NeuralNode<16> layers2[16];
    struct NeuralNode<16> layers3[10];
};

extern "C" __global__ void img_result(unsigned char* images, const char* labels, float* network, float* costs) {
    int i = threadIdx.x;

    unsigned char* image = images + (i * 784);
    unsigned char label = labels[i];

    struct NeuralNetwork nn;

    for (int i = 0; i < 16; i++) {
        nn.layers1[i].bias = network[i * 785];
        for (int j = 0; j < 784; j++) {
            nn.layers1[i].weights[j] = network[1 + (i * 785) + j];
        }
    }
    for (int i = 0; i < 16; i++) {
        nn.layers2[i].bias = network[i * 17];
        for (int j = 0; j < 16; j++) {
            nn.layers2[i].weights[j] = network[1 + (i * 17) + j];
        }
    }

    for (int i = 0; i < 10; i++) {
        nn.layers3[i].bias = network[i * 17];
        for (int j = 0; j < 16; j++) {
            nn.layers3[i].weights[j] = network[1 + (i * 17) + j];
        }
    }

    // TODO: Rewrite run_calc from main.rs to C code here
    float* layer1_values = (float*) malloc(16 * sizeof(float));
    float* layer2_values = (float*) malloc(16 * sizeof(float));
    float* layer3_values = (float*) malloc(10 * sizeof(float));

    for (int n = 0; n < 16; ++n) {
        float sum = nn.layers1[n].bias;
        for (int j = 0; j < 784; ++j) {
            sum += nn.layers1[n].weights[j] * image[j];
        }
        layer1_values[n] = sum;
    }
    for (int n = 0; n < 16; ++n) {
        float sum = nn.layers2[n].bias;
        for (int j = 0; j < 16; ++j) {
            sum += nn.layers2[n].weights[j] * layer1_values[j];
        }
        layer2_values[n] = sum;
    }
    for (int n = 0; n < 10; ++n) {
        float sum = nn.layers3[n].bias;
        for (int j = 0; j < 16; ++j) {
            sum += nn.layers3[n].weights[j] * layer2_values[j];
        }
        layer3_values[n] = sum;
    }

    costs[i] = 0.0;
    for (unsigned int i = 0; i < 10; i++) {
        float val = pow((label == i) - layer3_values[i], 2);
        costs[i] += 1.0f / (1.0f + exp(-val));
    }
}