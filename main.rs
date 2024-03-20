use ndarray::{Array, Array1, Array2, ArrayView, ArrayView1, ArrayView2, Axis};
use rand::distributions::{Distribution, Uniform};
use std::io::Write;
use std::str::FromStr;
use std::{fs::File, io::Read};
use clap::Parser;

const NUM_FEATURES: usize = 784;
const LINE_SIZE: usize = 785;
const NUM_CLASSES: usize = 10;
const GREYSCALE_SIZE: f64 = 255f64;

/// Represents a neural net
struct NeuralNet {
    layers: Vec<(Array2<f64>, Array1<f64>)>, // Each layer holds a weight matrix and a bias vector
    num_epochs: usize,                       // Training hyperparams
    batch_size: usize,
    learning_rate: f64,
}

/// Represents the dataset 
struct Dataset {
    data: Array2<f64>,
    target: Array2<f64>, // Target labels in one-hot encoding
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The path of the training dataset
    #[arg(short, long)]
    train_path: String,
    
    /// The path of the validation dataset
    #[arg(short, long)]
    validation_path: String,

    /// Network structure, e.g. [784, 500, 300, 10]
    #[arg(short, long, value_parser, num_args = 2.., value_delimiter = ' ')]
    network_structure: Vec<usize>,

    /// Learning rate of the network
    #[arg(short, long, default_value_t = 0.01)]
    learning_rate: f64,

    /// Batch size of the network
    #[arg(short, long, default_value_t = 50)]
    batch_size: usize,

    /// Number of epochs to train the network for
    #[arg(short, long, default_value_t = 11)]
    num_epochs: usize,

    /// Debug mode (save loss in a "time     loss" format)
    #[arg(short, long, default_value = None)]
    debug_path: Option<String>,
}

impl NeuralNet {
    /// Construct a new neural net according to the specified hyperparams
    pub fn new(
        layer_structure: Vec<usize>,
        num_epochs: usize,
        batch_size: usize,
        learning_rate: f64,
    ) -> NeuralNet {
        let mut layers = vec![];
        let mut rng = rand::thread_rng();
        // Weights are initialized from a uniform distribiution
        let distribution = Uniform::new(-0.3, 0.3);

        for i in 0..layer_structure.len() - 1 {
            // Random matrix of the weights between this layer and the next layer
            let weights = Array::zeros((layer_structure[i], layer_structure[i + 1]))
                .map(|_: &f64| distribution.sample(&mut rng));
            // Bias vector between this layer and the next layer. Init'd to ondes
            let bias = Array::ones(layer_structure[i + 1]);

            layers.push((weights, bias));
        }

        NeuralNet {
            layers,
            num_epochs,
            batch_size,
            learning_rate,
        }
    }

    // Perform a forward pass of the network on some input.
    // Returns the outputs of the hidden layers, and the non-activated outputs of the hidden layers (used for backprop)
    fn forward(&self, inputs: &ArrayView2<f64>) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let mut hidden = vec![];
        let mut hidden_linear = vec![];
        // The first layer is a passthrough layer, so it outputs whatever its input is
        hidden.push(inputs.to_owned());

        // We iterate for every layer
        let mut it = self.layers.iter().peekable();

        // Iterate over the layers
        while let Some(layer) = it.next() {
            // The output of the layer without applying the activation function
            let lin_output = hidden.last().unwrap().dot(&layer.0) + &layer.1;
            // The real output of the layer - If the layer is a hidden layer, we apply the activation function
            // and otherwise (this is the output layer) the output is the same as the linear output
            let real_output = lin_output.map(|x| match it.peek() {
                Some(_) => relu(*x),
                None => *x,
            });

            hidden.push(real_output);
            hidden_linear.push(lin_output);
 
        }

        (hidden, hidden_linear)
    }

    /// Calculate the gradients using backprop and perform a GD step
    fn backward_and_update(
        &mut self,
        hidden: Vec<Array2<f64>>,
        hidden_linear: Vec<Array2<f64>>,
        grad: Array2<f64>,
    ) {
        // The gradient WRT the current layer
        let mut grad_help = grad;

        for idx in (0..self.layers.len()).rev() {
            // If we aren't at the last layer, we need to change the gradient
            if idx != self.layers.len() - 1 {
                let step_mat = hidden_linear[idx].map(|x| step(*x));
                grad_help = grad_help * step_mat;
            }

            // Gradient WRT the weights in the current layer
            let weight_grad = hidden[idx].t().dot(&grad_help);
            // Gradient WRT the biases in the current layer
            let bias_grad = &grad_help.mean_axis(Axis(0)).unwrap();

            // Perform GD step
            let new_weights = &self.layers[idx].0 - self.learning_rate * weight_grad;
            let new_biases = &self.layers[idx].1 - self.learning_rate * bias_grad;

            // Update the helper variable
            grad_help = grad_help.dot(&self.layers[idx].0.t());

            self.layers[idx] = (new_weights, new_biases);
        }
    }

    /// Predict the probabities for a set of instances - each instance is a row in "inputs"
    fn predict(&self, inputs: &ArrayView2<f64>) -> Array2<f64> {
        let (hidden, _) = self.forward(inputs);
        let scores = hidden.last().unwrap();
        // Construct the softmax
        let mut predictions = Array::zeros((0, scores.ncols()));

        for row in scores.axis_iter(Axis(0)) {
            predictions.push_row(softmax(row).view()).unwrap();
        }

        predictions
    }

    /// Fit the model to the dataset
    fn fit(&mut self, dataset: &Dataset, debug_path: &Option<String>) -> Option<Vec<(usize, f64)>> {
        // Used for writing the debug output
        let mut batch_cnt: usize = 0;
        let mut losses = vec![];
        let is_debug = !debug_path.is_none();

        for _ in 0..self.num_epochs {
            // Get a batch of instances and their targets
            for (input_batch, target_batch) in dataset
                .data
                .axis_chunks_iter(Axis(0), self.batch_size)
                .zip(dataset.target.axis_chunks_iter(Axis(0), self.batch_size))
            {
                let (hidden, hidden_linear) = self.forward(&input_batch);

                let scores = hidden.last().unwrap();
                let mut predictions = Array::zeros((0, scores.ncols()));

                // Construct softmax matrix
                for row in scores.axis_iter(Axis(0)) {
                    predictions.push_row(softmax(row).view()).unwrap();
                }

                // Push to the losses vector if we're in debug mode
                if is_debug {
                    let loss = cross_entropy(&predictions, target_batch);
                    losses.push((batch_cnt, loss));

                    batch_cnt += 1;
                }

                // Gradient is initialized to the gradient of the loss WRT the output layer
                let grad = predictions - target_batch;

                self.backward_and_update(hidden, hidden_linear, grad);
            }
        }

        match debug_path {
            Some(_) => Some(losses),
            None => None,
        }
    }
}

/// Parse a record (e.g. CSV record) of the form <x1><sep><x2><sep>...
/// Returns a vector of the xi's if the function was succesful
/// and None otherwise
fn parse_line<T: FromStr>(s: &str, seperator: char) -> Option<Vec<T>> {
    let mut record = Vec::<T>::new();

    for x in s.split(seperator) {
        match T::from_str(x) {
            Ok(val) => {
                record.push(val);
            }
            _ => return None,
        }
    }

    Some(record)
}

/// Parse a line in the dataset. Return the pixels and the label
/// Line is stored in the format: <label>,<pixel0x0>,<pixel0x1>,...
/// The dataset is taken from here https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
fn parse_dataset_line(line: &str) -> Option<(Vec<f64>, f64)> {
    match parse_line(line, ',') {
        Some(v) => match v.len() {
            // we divide by 255 to normalize
            LINE_SIZE => Some((
                v[1..LINE_SIZE].iter().map(|x| x / GREYSCALE_SIZE).collect(),
                v[0],
            )),
            _ => None,
        },
        _ => None,
    }
}

// Return matrix that represents the dataset
fn parse_dataset(path: &str) -> Dataset {
    let file = File::open(path);
    let mut data = Array::zeros((0, NUM_FEATURES));
    let mut target = Array::zeros((0, NUM_CLASSES));
    let mut contents = String::new();

    file.unwrap().read_to_string(&mut contents).unwrap();

    for line in contents.lines().skip(1).take_while(|x| !x.is_empty()) {
        let line = parse_dataset_line(line).unwrap();
        let pixels = line.0;
        let label = line.1 as usize;
        // Construct one-hot encoding for the label
        let one_hot_target: Vec<f64> = (0..NUM_CLASSES)
            .map(|idx| if idx == label { 1f64 } else { 0f64 })
            .collect();

        data.push_row(ArrayView::from(&pixels)).unwrap();
        target.push_row(ArrayView::from(&one_hot_target)).unwrap();
    }

    Dataset { data, target }
}

/// Activation function
fn relu(z: f64) -> f64 {
    z.max(0f64)
}

/// Softmax function - Convert scores into a probability distribution
fn softmax(scores: ArrayView1<f64>) -> Array1<f64> {
    let max = scores.iter().max_by(|x, y| x.total_cmp(y)).unwrap();
    // We use a numerical trick where we shift the elements by the max, because otherwise
    // We would have to compute the exp of very large values which wraps to NaN
    let shift_scores = scores.map(|x| x - max);
    let sum: f64 = shift_scores.iter().map(|x| x.exp()).sum();

    (0..scores.len())
        .map(|x| shift_scores[x].exp() / sum)
        .collect()
}

/// Derivative of ReLU
fn step(z: f64) -> f64 {
    if z >= 0f64 {
        1f64
    } else {
        0f64
    }
}

/// Calculate the cross-entropy loss on a given batch
fn cross_entropy(actual: &Array2<f64>, target: ArrayView2<f64>) -> f64 {
    let total: f64 = actual
        .axis_iter(Axis(0))
        .zip(target.axis_iter(Axis(0)))
        .map(|(actual_row, target_row)| target_row.dot(&actual_row.map(|x| x.log2())))
        .sum();

    -1f64 * (1f64 / actual.nrows() as f64) * total
}

/// Test the model on the validation set
fn test_model(path: &str, model: &NeuralNet) {
    let dataset = parse_dataset(path);
    let predictions = model.predict(&dataset.data.view());

    let mut num_mistakes = 0;

    for (prediction, target_row) in predictions
        .axis_iter(Axis(0))
        .zip(dataset.target.axis_iter(Axis(0)))
    {
        let predict_digit = prediction
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0;
        let actual_digit = target_row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0;

        if predict_digit != actual_digit {
            num_mistakes += 1;
        }
    }

    println!("The number of mistakes is {}", num_mistakes);
}

/// Write the losses to a debug file
fn write_losses(debug_path: &str, losses: Vec<(usize, f64)>) -> std::io::Result<()> {
    let mut file = File::create(debug_path)?;

    for (x, y) in losses {
        file.write_all(format!("{}    {}\n", x, y).as_bytes())?;
    }

    Ok(())
}

fn main() {
    let args = Args::parse();
    
    let dataset = parse_dataset(&args.train_path);
    let mut neural_net = NeuralNet::new(args.network_structure, args.num_epochs, args.batch_size, args.learning_rate);

    let losses = neural_net.fit(&dataset, &args.debug_path);

    if !losses.is_none() {
        let _ = write_losses(&args.debug_path.unwrap(), losses.unwrap());
    }

    test_model(
        &args.validation_path,
        &neural_net,
    );
}
