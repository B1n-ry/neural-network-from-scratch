use std::{array::from_fn, error::Error, f32::consts::E, path::Path};
use cudarc::{driver::{LaunchAsync, LaunchConfig}, nvrtc::Ptx};
use rand::{rngs::ThreadRng, thread_rng, Rng};
use idx_parsing::IdxFile;


mod idx_parsing;

const N_THREADS: usize = 200;

struct NeuralNode<const N: usize> {
    weights: [f32; N],
    bias: f32,
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

    fn run_calc(&self, input: &Vec<f32>) -> Vec<f32> {
        let mut node_values: (Vec<f32>, Vec<f32>, Vec<f32>) = (
            vec![0.0; self.layers.0.len()],
            vec![0.0; self.layers.1.len()],
            vec![0.0; self.layers.2.len()],
        );

        for (node, val) in &mut self.layers.0.iter().zip(&mut node_values.0) {
            *val = input.iter()
                .zip(node.weights.iter())
                .map(|(i, w)| i * w)
                .sum::<f32>();
            *val += node.bias;
    
            *val = signmoid(*val);
        }
        for (node, val) in &mut self.layers.1.iter().zip(&mut node_values.1) {
            *val = node_values.0.iter()
                .zip(node.weights.iter())
                .map(|(v, w)| v * w)
                .sum::<f32>();
            *val += node.bias;
    
            *val = signmoid(*val);
        }
        for (node, val) in &mut self.layers.2.iter().zip(&mut node_values.2) {
            *val = node_values.1.iter()
                .zip(node.weights.iter())
                .map(|(v, w)| v * w)
                .sum::<f32>();
            *val += node.bias;
    
            *val = signmoid(*val);
        }

        node_values.2
    }

    fn calc_cost(&self, correct: u8, net_result: &Vec<f32>) -> f32 {
        let expected: [f32; 10] = from_fn(|i| i).map(|v| (v as u8 == correct) as u8 as f32);
    
        let sum = expected.iter()
            .zip(net_result.iter())
            .map(|(e, r)| (e - r).powi(2))
            .sum::<f32>();

        sum
    }

    fn as_vec(&self) -> Vec<f32> {
        let mut vector = vec![];

        for node in &self.layers.0 {
            vector.push(node.bias);

            for w in node.weights {
                vector.push(w);
            }
        }
        for node in &self.layers.1 {
            vector.push(node.bias);

            for w in node.weights {
                vector.push(w);
            }
        }
        for node in &self.layers.2 {
            vector.push(node.bias);

            for w in node.weights {
                vector.push(w);
            }
        }

        vector
    }
}

fn signmoid(i: f32) -> f32 {
    1.0 / (1.0 + E.powf(-i))
}

fn run_once(image: &Vec<u8>, label: u8, network: &NeuralNetwork) -> f32 {
    let mut input: Vec<f32> = vec![];

    for &d in image {
        input.push(d as f32 / 255.0);
    }

    let result = network.run_calc(&input);

    network.calc_cost(label, &result)
}

fn cuda_parallelize(images: Vec<u8>, labels: Vec<u8>, network: &NeuralNetwork, dataset_size: usize) -> Result<f32, Box<dyn Error>> {
    let gpu = cudarc::driver::CudaDevice::new(0)?;

    // allocate buffers
    let image_slice = gpu.htod_copy(images[0..dataset_size].to_vec())?;
    let label_slice = gpu.htod_copy(labels[0..dataset_size].to_vec())?;

    let network_vec = network.as_vec();
    let network_slice = gpu.htod_copy(network_vec)?;
    let mut out = gpu.alloc_zeros::<f32>(dataset_size)?;

    let ptx_file = Ptx::from_file("./cuda/img_result.ptx");
    gpu.load_ptx(ptx_file, "neural01", &["img_result"])?;

    let img_result = gpu.get_func("neural01", "img_result").unwrap();
    let config = LaunchConfig::for_num_elems(dataset_size as u32);
    unsafe { img_result.launch(config, (&image_slice, &label_slice, &network_slice, &mut out, dataset_size)) }?;

    let costs: Vec<f32> = gpu.dtoh_sync_copy(&out)?;
    
    let avg = costs.iter().sum::<f32>() / dataset_size as f32;

    Ok(avg)
}

fn main() {
    let dataset_size = 40_000;
    
    let images: IdxFile = IdxFile::load("mnist/train-images.idx3-ubyte");
    let labels: IdxFile = IdxFile::load("mnist/train-labels.idx1-ubyte");
    
    let network: NeuralNetwork = NeuralNetwork::random();
    
    let res = cuda_parallelize(images.data, labels.data, &network, dataset_size);

    println!("{}", res.unwrap());

    /* let chunk_size = dataset_size / N_THREADS;

    let mut all_data: Vec<(f32, Vec<u8>, u8)> = izip!(
        vec![0.0; dataset_size],
        (0..dataset_size).map(|i| images.get_image(i)),
        (0..dataset_size).map(|i| labels.get_byte(i))
    ).collect();

    all_data.par_chunks_mut(chunk_size).for_each(|slice| {
        for s in slice {
            s.0 = run_once(&s.1, s.2, &network);
        }
    });
    
    let avg_cost = all_data.iter().map(|t| t.0).sum::<f32>() / dataset_size as f32;

    println!("Avg: {}", avg_cost); */
}