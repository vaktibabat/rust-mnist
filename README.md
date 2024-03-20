# rust-mnist
Making a Neural Net for Digit Recognition from scratch in Rust for a blogpost.
The training hyperparameters can be adjusted. 
Example usage:
`./rust_neuralnet --train-path path_to_train_set --validation-path path_to_validation_set -n 784, 128, 10` (n specifies the network architecture)

Compile with `cargo build --release`
