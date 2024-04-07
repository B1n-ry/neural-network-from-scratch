use std::{array::from_fn, f32::consts::E, time::SystemTime};

use rand::{rngs::ThreadRng, thread_rng, Rng};
use idx_parsing::IdxFile;

use rayon::prelude::*;

mod idx_parsing;


const N_THREADS: usize = 200;



#[derive(Clone, Copy)]
struct NeuralNode<const N: usize> {
    weights: [f32; N],
    bias: f32,
    value: f32,
}
impl<const N: usize> NeuralNode<N> {
    fn with_random_weights_and_bias(rng: &mut ThreadRng) -> NeuralNode<N> {

        let mut weights = [0.0; N];
        for i in 0..N {
            weights[i] = rng.gen_range(-1.0..1.0);
        }

        let limit = weights.len() as f32;

        NeuralNode {
            weights: weights,
            bias: rng.gen_range(-limit..limit),
            value: 0.0,
        }
    }
}


struct NeuralNetwork {
    layers: (
        [NeuralNode<784>; 16],
        [NeuralNode<16>; 16],
        [NeuralNode<16>; 10],
    )
}
impl NeuralNetwork {
    fn random() -> NeuralNetwork {
        let mut rng = thread_rng();
        
        NeuralNetwork {
            layers: (
                [(); 16].map(|_| NeuralNode::with_random_weights_and_bias(&mut rng)),
                [(); 16].map(|_| NeuralNode::with_random_weights_and_bias(&mut rng)),
                [(); 10].map(|_| NeuralNode::with_random_weights_and_bias(&mut rng)),
            )
        }
    }

    fn run_calc(&mut self, input: &Vec<f32>) {
        for node in &mut self.layers.0 {
            node.value = input.iter()
                .zip(node.weights.iter())
                .map(|(i, w)| i * w)
                .sum::<f32>();
            node.value += node.bias;
    
            node.value = signmoid(node.value);
        }
        for node in &mut self.layers.1 {
            node.value = self.layers.0.iter()
                .zip(node.weights.iter())
                .map(|(n, w)| n.value * w)
                .sum::<f32>();
            node.value += node.bias;
    
            node.value = signmoid(node.value);
        }
        for node in &mut self.layers.2 {
            node.value = self.layers.1.iter()
                .zip(node.weights.iter())
                .map(|(n, w)| n.value * w)
                .sum::<f32>();
            node.value += node.bias;
    
            node.value = signmoid(node.value);
        }
    }
    fn calc_cost(&self, correct: u8) -> f32 {
        let result: [f32; 10] = self.layers.2.map(|n| n.value);
        let expected: [f32; 10] = from_fn(|i| i).map(|v| (v as u8 == correct) as u8 as f32);
    
        let sum = expected.iter()
            .zip(result.iter())
            .map(|(e, r)| (e - r).powi(2))
            .sum::<f32>();

        sum
    }
}

fn signmoid(i: f32) -> f32 {
    1.0 / (1.0 + E.powf(-i))
}

fn run_slice(images: &IdxFile, labels: &IdxFile, costs: &mut [f32]) {
    for i in 0..costs.len() {
        let mut input: Vec<f32> = vec![];

        for d in images.get_image(i) {
            input.push(d as f32 / 255.0);
        }

        let correct = labels.get_byte(i);

        let mut network: NeuralNetwork = NeuralNetwork::random();

        network.run_calc(&input);

        let cost = network.calc_cost(correct);
        
        costs[i] = cost;
    }
}

fn main() {
    let images: IdxFile = IdxFile::load("mnist/train-images.idx3-ubyte");
    let labels: IdxFile = IdxFile::load("mnist/train-labels.idx1-ubyte");

    let mut costs: Vec<f32> = vec![0.0; labels.data.len()];
    let chunk_size = costs.len() / N_THREADS;

    costs.par_chunks_mut(chunk_size).for_each(|slice| {
        run_slice(&images, &labels, slice);
    });

    let avg_cost = costs.iter().sum::<f32>() / costs.len() as f32;

    println!("Avg: {}", avg_cost);
}