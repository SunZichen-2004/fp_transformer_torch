#!/usr/bin/env python3
"""
Maximum likelihood training for an unbiased free energy model F_theta(s) from
multi-lambda dihedral angle samples using JAX.
"""

from __future__ import annotations

import argparse
import pickle
import re
from glob import glob
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax


LAMBDA_PATTERN = re.compile(r"lambda(\d+p\d+)", re.IGNORECASE)


def parse_lambda_from_filename(file_path: str | Path) -> float:
    """
    Extract lambda from a filename segment like 'lambda2p000000' -> 2.000000.
    The letter 'p' is treated as the decimal point.
    """
    name = Path(file_path).name
    match = LAMBDA_PATTERN.search(name)
    if match is None:
        raise ValueError(
            f"Could not parse lambda from filename '{name}'. "
            f"Expected a substring matching 'lambda<number>p<fraction>', e.g. lambda2p000000."
        )
    numeric_token = match.group(1).replace("p", ".", 1)
    return float(numeric_token)


def load_all_dihedral_datasets(
    data_directory: Path,
    expected_num_dihedral_angles: int | None = None,
) -> tuple[list[np.ndarray], list[float], list[int], int]:
    """
    Scan data_directory for *_dihedral.npy files.

    Returns:
        samples_per_dataset: list of arrays, each shaped [N_k, D] in radians (D = num dihedrals).
        lambda_per_dataset: list of lambda values (one per file).
        sample_counts: list of N_k.
        num_dihedral_angles: D (same for every file).
    """
    pattern = str(data_directory / "*_dihedral*")
    file_paths = sorted(glob(pattern))
    if len(file_paths) == 0:
        raise FileNotFoundError(
            f"No files matching '*_dihedral.npy' were found under directory: {data_directory.resolve()}\n"
            f"Expected names like 'stage0000_lambda2p000000_dihedral.npy'."
        )

    samples_per_dataset: list[np.ndarray] = []
    lambda_per_dataset: list[float] = []
    sample_counts: list[int] = []
    num_dihedral_angles: int | None = None

    for file_path in file_paths:
        lambda_value = parse_lambda_from_filename(file_path)
        raw_array = np.load(file_path, mmap_mode="r")
        array = np.asarray(raw_array, dtype=np.float64)

        if array.size == 0:
            raise ValueError(f"File '{file_path}' is empty.")

        if array.ndim == 3 and array.shape[-1] == 1:
            array = array.reshape(array.shape[0], array.shape[1])
        if array.ndim != 2:
            array = array.reshape(-1, array.shape[-1])

        dihedral_count = int(array.shape[-1])
        if dihedral_count < 1:
            raise ValueError(f"File '{file_path}' has invalid shape {array.shape}.")
        if num_dihedral_angles is None:
            num_dihedral_angles = dihedral_count
            if expected_num_dihedral_angles is not None and num_dihedral_angles != expected_num_dihedral_angles:
                raise ValueError(
                    f"File '{file_path}' has {num_dihedral_angles} dihedral angles, "
                    f"but --num_dihedral_angles {expected_num_dihedral_angles} was requested."
                )
        elif dihedral_count != num_dihedral_angles:
            raise ValueError(
                f"File '{file_path}' has last dimension {dihedral_count}, "
                f"expected {num_dihedral_angles} (must match all other files)."
            )

        angles = array.reshape(-1, num_dihedral_angles)
        # Map to [-pi, pi) for consistency (periodic variables).
        angles_wrapped = (angles + np.pi) % (2 * np.pi) - np.pi

        samples_per_dataset.append(angles_wrapped.astype(np.float64))
        lambda_per_dataset.append(lambda_value)
        sample_counts.append(int(angles_wrapped.shape[0]))

    assert num_dihedral_angles is not None
    return samples_per_dataset, lambda_per_dataset, sample_counts, num_dihedral_angles


def dihedral_angles_to_sincos_features(dihedral_angles: jnp.ndarray) -> jnp.ndarray:
    """
    Map angles phi_j to [sin(phi_1), cos(phi_1), ..., sin(phi_D), cos(phi_D)].
    dihedral_angles: [..., D] -> features [..., 2*D]
    """
    num_dihedral_angles = dihedral_angles.shape[-1]
    if num_dihedral_angles < 1:
        raise ValueError(f"dihedral_angles last dimension must be >= 1, got shape {dihedral_angles.shape}.")
    sine_components = jnp.sin(dihedral_angles)
    cosine_components = jnp.cos(dihedral_angles)
    stacked = jnp.stack([sine_components, cosine_components], axis=-1)
    new_shape = dihedral_angles.shape[:-1] + (2 * num_dihedral_angles,)
    return stacked.reshape(new_shape)


def initialize_multilayer_perceptron_parameters(
    random_key: jax.Array,
    input_dimension: int,
    hidden_layer_sizes: tuple[int, ...],
    output_dimension: int,
):
    """
    Xavier-style init for an MLP; returns a nested dict of weights and biases.
    """
    layer_widths = (input_dimension,) + hidden_layer_sizes + (output_dimension,)
    keys = jax.random.split(random_key, len(layer_widths) - 1)
    layers = []
    for layer_index, subkey in enumerate(keys):
        fan_in = layer_widths[layer_index]
        fan_out = layer_widths[layer_index + 1]
        weight_matrix = jax.random.normal(subkey, (fan_in, fan_out)) * jnp.sqrt(2.0 / fan_in)
        bias_vector = jnp.zeros((fan_out,))
        layers.append({"weight_matrix": weight_matrix, "bias_vector": bias_vector})
    return {"layers": layers}


def forward_free_energy_model(parameters, input_features: jnp.ndarray) -> jnp.ndarray:
    """
    Scalar free energy F_theta(s) per row. input_features: [batch, 2*D].
    Returns shape [batch].
    """
    hidden = input_features
    num_layers = len(parameters["layers"])
    for layer_index in range(num_layers - 1):
        weights = parameters["layers"][layer_index]["weight_matrix"]
        biases = parameters["layers"][layer_index]["bias_vector"]
        hidden = jax.nn.silu(hidden @ weights + biases)
    final_weights = parameters["layers"][-1]["weight_matrix"]
    final_biases = parameters["layers"][-1]["bias_vector"]
    output = hidden @ final_weights + final_biases
    return output.reshape(-1)


def compute_log_partition_function_batch(
    free_energy_on_reference_points: jnp.ndarray,
    alpha_values_per_dataset: jnp.ndarray,
    log_hypercube_volume: jnp.ndarray,
    num_reference_points: int,
) -> jnp.ndarray:
    """
    For each dataset k, log Z_k = log(V) + log(mean_m exp(-alpha_k * F(s_m))) with log-sum-exp.

    free_energy_on_reference_points: [M]
    alpha_values_per_dataset: [K]
    Returns log_partition_values: [K]
    """
    log_reference_count = jnp.log(jnp.asarray(num_reference_points, dtype=jnp.float32))

    def log_partition_for_alpha(alpha_k: jnp.ndarray) -> jnp.ndarray:
        scaled = -alpha_k * free_energy_on_reference_points
        return log_hypercube_volume + jax.scipy.special.logsumexp(scaled) - log_reference_count

    return jax.vmap(log_partition_for_alpha)(alpha_values_per_dataset)


def negative_log_likelihood_loss(
    parameters,
    all_sample_features: jnp.ndarray,
    dataset_index_per_sample: jnp.ndarray,
    alpha_values_per_dataset: jnp.ndarray,
    reference_dihedral_angles: jnp.ndarray,
    num_dihedral_angles: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Mean over all samples of ( alpha_k * F(s) + log Z_k ).

    Returns:
        total_loss: scalar (mean NLL)
        mean_alpha_times_free_energy: scalar
        mean_log_partition: scalar (mean of log Z_k over samples, same weighting as loss)
    """
    log_hypercube_volume = jnp.asarray(num_dihedral_angles, dtype=jnp.float32) * jnp.log(
        2.0 * jnp.pi
    )
    num_reference_points = reference_dihedral_angles.shape[0]

    reference_features = dihedral_angles_to_sincos_features(reference_dihedral_angles)
    free_energy_on_reference = forward_free_energy_model(parameters, reference_features)

    log_partition_per_dataset = compute_log_partition_function_batch(
        free_energy_on_reference,
        alpha_values_per_dataset,
        log_hypercube_volume,
        num_reference_points,
    )

    free_energy_on_data = forward_free_energy_model(parameters, all_sample_features)
    alpha_per_sample = alpha_values_per_dataset[dataset_index_per_sample]
    log_partition_per_sample = log_partition_per_dataset[dataset_index_per_sample]

    per_sample_negative_log_likelihood = (
        alpha_per_sample * free_energy_on_data + log_partition_per_sample
    )
    total_loss = jnp.mean(per_sample_negative_log_likelihood)
    mean_alpha_times_free_energy = jnp.mean(alpha_per_sample * free_energy_on_data)
    mean_log_partition = jnp.mean(log_partition_per_sample)
    return total_loss, mean_alpha_times_free_energy, mean_log_partition


def build_concatenated_training_arrays(
    samples_per_dataset: list[np.ndarray],
    lambda_per_dataset: list[float],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Stack all samples into one array and record dataset index and alpha for each row.
    """
    alpha_values = jnp.array(
        [lam / (1.0 + lam) for lam in lambda_per_dataset], dtype=jnp.float32
    )
    feature_blocks = []
    index_blocks = []
    for dataset_index, samples in enumerate(samples_per_dataset):
        angles = jnp.asarray(samples, dtype=jnp.float32)
        feats = dihedral_angles_to_sincos_features(angles)
        feature_blocks.append(feats)
        index_blocks.append(
            jnp.full((feats.shape[0],), dataset_index, dtype=jnp.int32)
        )

    all_sample_features = jnp.concatenate(feature_blocks, axis=0)
    dataset_index_per_sample = jnp.concatenate(index_blocks, axis=0)
    return all_sample_features, dataset_index_per_sample, alpha_values


def main() -> None:
    parser = argparse.ArgumentParser(description="Train F_theta(s) by MLE on multi-lambda dihedral data.")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing *_dihedral.npy files.",
    )
    parser.add_argument("--steps", type=int, default=2000, help="Number of optimization steps.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--mc_samples",
        type=int,
        default=4096,
        help="Number of uniform reference points in [-pi, pi)^D for each Z_k estimate.",
    )
    parser.add_argument(
        "--num_dihedral_angles",
        type=int,
        default=None,
        help="Optional: expected number of dihedral angles D per sample (must match data).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=("adamw", "adam"),
        help="Optimizer (default: adamw).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay (ignored for adam).",
    )
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=[32, 32, 32],
        help="MLP hidden layer widths (default: 32 32 32).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=".",
        help="Directory for params.pkl and loss_history.npy.",
    )
    parser.add_argument(
        "--params_name",
        type=str,
        default="params.pkl",
        help="Filename for the saved parameter pickle inside --save_dir.",
    )
    parser.add_argument(
        "--loss_history_name",
        type=str,
        default="loss_history.npy",
        help="Filename for the saved loss history inside --save_dir.",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=100,
        help="Print diagnostics every this many steps.",
    )
    args = parser.parse_args()

    data_directory = Path(args.data_dir)
    save_directory = Path(args.save_dir)
    save_directory.mkdir(parents=True, exist_ok=True)

    samples_per_dataset, lambda_per_dataset, sample_counts, num_dihedral_angles = (
        load_all_dihedral_datasets(data_directory, args.num_dihedral_angles)
    )
    num_datasets = len(samples_per_dataset)
    total_sample_count = sum(sample_counts)

    print(
        f"Loaded {num_datasets} dataset(s), {total_sample_count} total samples, "
        f"D={num_dihedral_angles} dihedral angle(s) per sample."
    )
    for idx, (lam, nk) in enumerate(zip(lambda_per_dataset, sample_counts)):
        alpha = lam / (1.0 + lam)
        print(f"  dataset {idx}: lambda={lam:g}, alpha={alpha:.6f}, N={nk}")

    all_sample_features, dataset_index_per_sample, alpha_values_per_dataset = (
        build_concatenated_training_arrays(samples_per_dataset, lambda_per_dataset)
    )

    random_key = jax.random.PRNGKey(args.seed)
    random_key, init_key = jax.random.split(random_key)
    input_feature_dimension = 2 * num_dihedral_angles
    hidden_layer_sizes = tuple(int(w) for w in args.hidden_sizes)
    if len(hidden_layer_sizes) < 1 or any(w < 1 for w in hidden_layer_sizes):
        raise ValueError(f"--hidden_sizes must be positive ints, got {args.hidden_sizes}")
    parameters = initialize_multilayer_perceptron_parameters(
        init_key,
        input_dimension=input_feature_dimension,
        hidden_layer_sizes=hidden_layer_sizes,
        output_dimension=1,
    )
    print(f"MLP hidden_layer_sizes={hidden_layer_sizes}")

    if args.optimizer == "adamw":
        optimizer = optax.adamw(learning_rate=args.lr, weight_decay=args.weight_decay)
        print(f"Using AdamW (lr={args.lr:g}, weight_decay={args.weight_decay:g})")
    else:
        optimizer = optax.adam(args.lr)
        print(f"Using Adam (lr={args.lr:g})")
    optimizer_state = optimizer.init(parameters)

    @jax.jit
    def training_step(
        parameters,
        optimizer_state,
        all_sample_features,
        dataset_index_per_sample,
        alpha_values_per_dataset,
        reference_dihedral_angles,
    ):
        def loss_wrapper(params):
            total_loss, mean_alpha_f, mean_logz = negative_log_likelihood_loss(
                params,
                all_sample_features,
                dataset_index_per_sample,
                alpha_values_per_dataset,
                reference_dihedral_angles,
                num_dihedral_angles,
            )
            return total_loss, (mean_alpha_f, mean_logz)

        (loss, (mean_alpha_f, mean_logz)), gradients = jax.value_and_grad(
            loss_wrapper, has_aux=True
        )(parameters)
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, parameters)
        new_parameters = optax.apply_updates(parameters, updates)
        return new_parameters, optimizer_state, loss, mean_alpha_f, mean_logz

    loss_history: list[float] = []
    data_term_history: list[float] = []
    logz_history: list[float] = []

    for step in range(args.steps):
        random_key, subkey = jax.random.split(random_key)
        reference_dihedral_angles = jax.random.uniform(
            subkey,
            (args.mc_samples, num_dihedral_angles),
            minval=-jnp.pi,
            maxval=jnp.pi,
            dtype=jnp.float32,
        )

        parameters, optimizer_state, loss_value, mean_alpha_f, mean_logz = training_step(
            parameters,
            optimizer_state,
            all_sample_features,
            dataset_index_per_sample,
            alpha_values_per_dataset,
            reference_dihedral_angles,
        )

        loss_history.append(float(loss_value))
        data_term_history.append(float(mean_alpha_f))
        logz_history.append(float(mean_logz))

        if step % args.log_every == 0 or step == args.steps - 1:
            print(
                f"step {step:5d} | loss {float(loss_value):.6f} | "
                f"data_term(mean alpha*F) {float(mean_alpha_f):.6f} | "
                f"mean_logZ {float(mean_logz):.6f}"
            )

    parameters_numpy = jax.tree_util.tree_map(lambda x: np.asarray(x), parameters)
    params_path = save_directory / args.params_name
    with open(params_path, "wb") as file_handle:
        pickle.dump(
            {
                "parameters": parameters_numpy,
                "hidden_layer_sizes": hidden_layer_sizes,
                "num_dihedral_angles": num_dihedral_angles,
                "input_feature_dimension": input_feature_dimension,
                "lambda_per_dataset": lambda_per_dataset,
                "alpha_per_dataset": [lam / (1.0 + lam) for lam in lambda_per_dataset],
                "seed": int(args.seed),
                "optimizer": str(args.optimizer),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay) if args.optimizer == "adamw" else 0.0,
                "steps": int(args.steps),
                "mc_samples": int(args.mc_samples),
            },
            file_handle,
        )
    print(f"Saved parameters to {params_path.resolve()}")

    history_array = np.stack(
        [np.array(loss_history), np.array(data_term_history), np.array(logz_history)],
        axis=0,
    )
    history_path = save_directory / args.loss_history_name
    np.save(history_path, history_array)
    print(
        f"Saved loss history (rows: total_loss, data_term, mean_logZ) to {history_path.resolve()}"
    )


if __name__ == "__main__":
    main()
