#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CKA (Centered Kernel Alignment) based model coherence testing.

This module provides tools to test the coherence between multiple models
using CKA similarity metrics on their embeddings/representations.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import json
import warnings

from contrastive.utils.logs import set_file_logger

log = set_file_logger(__file__)


def linear_CKA(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute linear CKA (Centered Kernel Alignment) between two feature matrices.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features_x)
        First representation (rows = samples).
    Y : np.ndarray of shape (n_samples, n_features_y)
        Second representation (rows = samples).

    Returns
    -------
    cka : float
        Similarity in [0, 1]. Higher values indicate more similar representations.

    Notes
    -----
    Linear CKA measures the similarity between two feature representations
    by comparing their centered Gram matrices. It is invariant to orthogonal
    transformations and isotropic scaling.

    References
    ----------
    Kornblith et al. "Similarity of Neural Network Representations Revisited"
    ICML 2019. https://arxiv.org/abs/1905.00414
    """
    # Validate inputs
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have the same number of samples. "
            f"Got X.shape={X.shape}, Y.shape={Y.shape}"
        )

    if X.shape[0] < 2:
        raise ValueError(
            f"Need at least 2 samples to compute CKA. Got {X.shape[0]} samples."
        )

    # Center each feature (column-wise)
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)

    # Cross-covariance
    cross_cov = Xc.T @ Yc

    # Frobenius norms
    numerator = np.linalg.norm(cross_cov, 'fro') ** 2
    denom = (np.linalg.norm(Xc.T @ Xc, 'fro') *
             np.linalg.norm(Yc.T @ Yc, 'fro'))

    return numerator / denom if denom > 0 else 0.0


def load_embeddings_pt(
    pt_path: str,
    embeddings_key: str = 'embeddings',
    subject_key: str = 'subject_ids',
) -> Tuple[None, np.ndarray, List[str]]:
    """
    Load embeddings from a PyTorch .pt file.

    Parameters
    ----------
    pt_path : str
        Path to the .pt file. The file may contain either:
        - A plain tensor of shape (n_samples, n_features), or
        - A dict with at least an ``embeddings_key`` entry (tensor or array)
          and optionally a ``subject_key`` entry (list/tensor of IDs).
    embeddings_key : str, optional
        Dict key for the embedding tensor when the file contains a dict.
        Default is ``'embeddings'``.
    subject_key : str, optional
        Dict key for subject IDs when the file contains a dict.
        Default is ``'subject_ids'``.

    Returns
    -------
    data : None
        Placeholder for API compatibility with :func:`load_embeddings_csv`
        (which returns a DataFrame). Always ``None`` for PT files.
    embeddings : np.ndarray
        Embedding matrix of shape (n_samples, n_features).
    subject_ids : list of str
        Subject IDs. Falls back to ``['0', '1', ...]`` when not present.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required to load .pt files: pip install torch") from exc

    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"PT file not found: {pt_path}")

    data = torch.load(pt_path, map_location='cpu', weights_only=False)

    if isinstance(data, dict):
        if embeddings_key not in data:
            raise KeyError(
                f"Key '{embeddings_key}' not found in {pt_path}. "
                f"Available keys: {list(data.keys())}"
            )
        raw_embeddings = data[embeddings_key]
        raw_ids = data.get(subject_key, None)
    else:
        raw_embeddings = data
        raw_ids = None

    # Convert to numpy
    if hasattr(raw_embeddings, 'numpy'):
        embeddings = raw_embeddings.detach().float().numpy()
    else:
        embeddings = np.asarray(raw_embeddings, dtype=np.float32)

    if raw_ids is not None:
        if hasattr(raw_ids, 'tolist'):
            subject_ids = [str(x) for x in raw_ids.tolist()]
        else:
            subject_ids = [str(x) for x in raw_ids]
    else:
        subject_ids = [str(i) for i in range(len(embeddings))]

    log.info(
        f"Loaded embeddings from {pt_path}: "
        f"shape={embeddings.shape}, n_subjects={len(subject_ids)}"
    )

    return None, embeddings, subject_ids


def load_embeddings(
    path: str,
    subject_column: str = 'ID',
    embeddings_key: str = 'embeddings',
    subject_key: str = 'subject_ids',
) -> Tuple[Optional[pd.DataFrame], np.ndarray, List[str]]:
    """
    Load embeddings from a CSV or PT file, dispatching on the file extension.

    Parameters
    ----------
    path : str
        Path to a ``.csv`` or ``.pt`` file.
    subject_column : str, optional
        Column name for subject IDs in CSV files.
    embeddings_key : str, optional
        Dict key for embeddings in PT files.
    subject_key : str, optional
        Dict key for subject IDs in PT files.

    Returns
    -------
    Same as :func:`load_embeddings_csv` / :func:`load_embeddings_pt`.
    """
    suffix = Path(path).suffix.lower()
    if suffix == '.pt':
        return load_embeddings_pt(path, embeddings_key=embeddings_key, subject_key=subject_key)
    elif suffix == '.csv':
        return load_embeddings_csv(path, subject_column=subject_column)
    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Expected .csv or .pt.")


def load_embeddings_csv(csv_path: str,
                        subject_column: str = 'ID') -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Load embeddings from a CSV file generated by embeddings_pipeline.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing embeddings.
    subject_column : str, optional
        Name of the column containing subject IDs. Default is 'ID'.

    Returns
    -------
    df : pd.DataFrame
        Full dataframe with all columns.
    embeddings : np.ndarray
        Embedding matrix of shape (n_subjects, n_features).
    subject_ids : list of str
        List of subject IDs corresponding to each row.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path, index_col=0)

    # Extract subject IDs
    if subject_column in df.columns:
        subject_ids = df[subject_column].tolist()
    elif subject_column in df.index.names or df.index.name == subject_column:
        subject_ids = df.index.tolist()
    else:
        log.warning(
            f"Subject column '{subject_column}' not found. Using index as subject IDs."
        )
        subject_ids = df.index.tolist()

    # Extract embedding columns (exclude labels and ID columns)
    exclude_cols = ['labels', 'label', subject_column]
    embedding_cols = [col for col in df.columns if col not in exclude_cols]

    if not embedding_cols:
        raise ValueError(f"No embedding columns found in {csv_path}")

    embeddings = df[embedding_cols].values

    log.info(
        f"Loaded embeddings from {csv_path}: "
        f"shape={embeddings.shape}, n_subjects={len(subject_ids)}"
    )

    return df, embeddings, subject_ids


def align_embeddings_by_subjects(
    embeddings_list: List[Tuple[np.ndarray, List[str]]],
    model_names: List[str]
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Align multiple embedding matrices to have the same subjects in the same order.

    Parameters
    ----------
    embeddings_list : list of tuples
        Each tuple is (embeddings, subject_ids) for one model.
    model_names : list of str
        Names of the models corresponding to embeddings_list.

    Returns
    -------
    aligned_embeddings : list of np.ndarray
        List of aligned embedding matrices.
    common_subjects : list of str
        List of common subject IDs in the aligned order.
    """
    if len(embeddings_list) != len(model_names):
        raise ValueError(
            f"embeddings_list and model_names must have the same length. "
            f"Got {len(embeddings_list)} vs {len(model_names)}"
        )

    # Find common subjects
    subject_sets = [set(subject_ids) for _, subject_ids in embeddings_list]
    common_subjects_set = set.intersection(*subject_sets)

    if not common_subjects_set:
        raise ValueError("No common subjects found across all models.")

    # Use the order from the first model
    common_subjects = [
        s for s in embeddings_list[0][1] if s in common_subjects_set
    ]

    log.info(
        f"Found {len(common_subjects)} common subjects across {len(model_names)} models"
    )

    # Align embeddings
    aligned_embeddings = []
    for (embeddings, subject_ids), model_name in zip(embeddings_list, model_names):
        # Create a mapping from subject_id to row index
        subject_to_idx = {s: i for i, s in enumerate(subject_ids)}

        # Reorder embeddings according to common_subjects
        aligned_indices = [subject_to_idx[s] for s in common_subjects]
        aligned_emb = embeddings[aligned_indices]
        aligned_embeddings.append(aligned_emb)

        log.debug(
            f"Aligned {model_name}: original shape={embeddings.shape}, "
            f"aligned shape={aligned_emb.shape}"
        )

    return aligned_embeddings, common_subjects


def compute_pairwise_cka(
    embeddings_list: List[np.ndarray],
    model_names: List[str]
) -> pd.DataFrame:
    """
    Compute pairwise CKA scores between all models.

    Parameters
    ----------
    embeddings_list : list of np.ndarray
        List of embedding matrices (must be aligned with same subjects).
    model_names : list of str
        Names of the models.

    Returns
    -------
    cka_matrix : pd.DataFrame
        Symmetric matrix of CKA scores with model names as index and columns.
    """
    n_models = len(embeddings_list)

    if n_models != len(model_names):
        raise ValueError(
            f"embeddings_list and model_names must have the same length. "
            f"Got {n_models} vs {len(model_names)}"
        )

    # Initialize CKA matrix
    cka_scores = np.zeros((n_models, n_models))

    # Compute pairwise CKA
    for i in range(n_models):
        for j in range(i, n_models):
            if i == j:
                cka_scores[i, j] = 1.0  # Self-similarity
            else:
                cka = linear_CKA(embeddings_list[i], embeddings_list[j])
                cka_scores[i, j] = cka
                cka_scores[j, i] = cka  # Symmetric

                log.debug(
                    f"CKA({model_names[i]}, {model_names[j]}) = {cka:.4f}"
                )

    # Create DataFrame
    cka_matrix = pd.DataFrame(
        cka_scores,
        index=model_names,
        columns=model_names
    )

    return cka_matrix


def compute_model_coherence_stats(cka_matrix: pd.DataFrame) -> Dict[str, float]:
    """
    Compute summary statistics from a CKA matrix.

    Parameters
    ----------
    cka_matrix : pd.DataFrame
        Symmetric matrix of CKA scores.

    Returns
    -------
    stats : dict
        Dictionary containing:
        - mean_cka: Mean CKA across all pairs (excluding diagonal)
        - median_cka: Median CKA across all pairs
        - std_cka: Standard deviation of CKA
        - min_cka: Minimum CKA score
        - max_cka: Maximum CKA score
        - mean_coherence: Same as mean_cka (alias for clarity)
    """
    # Extract upper triangle (excluding diagonal)
    n = len(cka_matrix)
    upper_triangle = []
    for i in range(n):
        for j in range(i + 1, n):
            upper_triangle.append(cka_matrix.iloc[i, j])

    upper_triangle = np.array(upper_triangle)

    stats = {
        'mean_cka': float(np.mean(upper_triangle)),
        'median_cka': float(np.median(upper_triangle)),
        'std_cka': float(np.std(upper_triangle)),
        'min_cka': float(np.min(upper_triangle)),
        'max_cka': float(np.max(upper_triangle)),
        'n_models': n,
        'n_pairs': len(upper_triangle)
    }
    stats['mean_coherence'] = stats['mean_cka']

    return stats


class CKACoherenceTester:
    """
    Test model coherence using CKA on embeddings from multiple models.

    This class provides a high-level interface for loading embeddings from
    multiple models and computing their pairwise CKA similarities.

    Parameters
    ----------
    embedding_paths : dict
        Dictionary mapping model names to embedding CSV paths.
        Example: {'model1': 'path/to/model1/embeddings.csv', ...}
    subject_column : str, optional
        Name of the column containing subject IDs. Default is 'ID'.

    Attributes
    ----------
    model_names : list of str
        List of model names.
    embeddings_dict : dict
        Dictionary storing loaded embeddings for each model.
    aligned_embeddings : list of np.ndarray
        List of aligned embedding matrices.
    common_subjects : list of str
        List of common subject IDs.
    cka_matrix : pd.DataFrame
        Pairwise CKA similarity matrix.
    """

    def __init__(self,
                 embedding_paths: Dict[str, str],
                 subject_column: str = 'ID',
                 embeddings_key: str = 'embeddings',
                 subject_key: str = 'subject_ids'):
        self.embedding_paths = embedding_paths
        self.subject_column = subject_column
        self.embeddings_key = embeddings_key
        self.subject_key = subject_key
        self.model_names = list(embedding_paths.keys())

        self.embeddings_dict = {}
        self.aligned_embeddings = None
        self.common_subjects = None
        self.cka_matrix = None

        log.info(f"Initialized CKACoherenceTester with {len(self.model_names)} models")

    def load_all_embeddings(self) -> None:
        """Load embeddings from all models."""
        log.info("Loading embeddings from all models...")

        for model_name, csv_path in self.embedding_paths.items():
            try:
                df, embeddings, subject_ids = load_embeddings(
                    csv_path,
                    subject_column=self.subject_column,
                    embeddings_key=self.embeddings_key,
                    subject_key=self.subject_key,
                )
                self.embeddings_dict[model_name] = {
                    'df': df,
                    'embeddings': embeddings,
                    'subject_ids': subject_ids
                }
                log.info(f"Loaded {model_name}: {embeddings.shape}")
            except Exception as e:
                log.error(f"Failed to load {model_name} from {csv_path}: {e}")
                raise

        log.info(f"Successfully loaded {len(self.embeddings_dict)} models")

    def align_embeddings(self) -> None:
        """Align embeddings across all models to have common subjects."""
        log.info("Aligning embeddings across models...")

        embeddings_list = [
            (self.embeddings_dict[name]['embeddings'],
             self.embeddings_dict[name]['subject_ids'])
            for name in self.model_names
        ]

        self.aligned_embeddings, self.common_subjects = align_embeddings_by_subjects(
            embeddings_list, self.model_names
        )

        log.info(
            f"Aligned embeddings: {len(self.common_subjects)} common subjects, "
            f"{len(self.aligned_embeddings)} models"
        )

    def compute_cka_matrix(self) -> pd.DataFrame:
        """
        Compute pairwise CKA matrix for all models.

        Returns
        -------
        cka_matrix : pd.DataFrame
            Symmetric matrix of CKA scores.
        """
        if self.aligned_embeddings is None:
            raise ValueError(
                "Must call align_embeddings() before computing CKA matrix"
            )

        log.info("Computing pairwise CKA scores...")

        self.cka_matrix = compute_pairwise_cka(
            self.aligned_embeddings,
            self.model_names
        )

        log.info("CKA matrix computed successfully")
        return self.cka_matrix

    def get_coherence_stats(self) -> Dict[str, float]:
        """
        Get summary statistics of model coherence.

        Returns
        -------
        stats : dict
            Dictionary of coherence statistics.
        """
        if self.cka_matrix is None:
            raise ValueError("Must call compute_cka_matrix() first")

        stats = compute_model_coherence_stats(self.cka_matrix)

        log.info(
            f"Coherence statistics: mean={stats['mean_cka']:.4f}, "
            f"std={stats['std_cka']:.4f}, range=[{stats['min_cka']:.4f}, {stats['max_cka']:.4f}]"
        )

        return stats

    def run_full_analysis(self) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Run the complete CKA coherence analysis pipeline.

        Returns
        -------
        cka_matrix : pd.DataFrame
            Pairwise CKA similarity matrix.
        stats : dict
            Summary statistics.
        """
        self.load_all_embeddings()
        self.align_embeddings()
        self.compute_cka_matrix()
        stats = self.get_coherence_stats()

        return self.cka_matrix, stats

    def save_results(self, output_dir: str) -> None:
        """
        Save CKA matrix and statistics to files.

        Parameters
        ----------
        output_dir : str
            Directory where results will be saved.
        """
        if self.cka_matrix is None:
            raise ValueError("Must run analysis before saving results")

        os.makedirs(output_dir, exist_ok=True)

        # Save CKA matrix
        cka_csv_path = os.path.join(output_dir, 'cka_matrix.csv')
        self.cka_matrix.to_csv(cka_csv_path)
        log.info(f"Saved CKA matrix to {cka_csv_path}")

        # Save statistics
        stats = self.get_coherence_stats()
        stats_json_path = os.path.join(output_dir, 'coherence_stats.json')
        with open(stats_json_path, 'w') as f:
            json.dump(stats, f, indent=2)
        log.info(f"Saved statistics to {stats_json_path}")

        # Save common subjects list
        subjects_txt_path = os.path.join(output_dir, 'common_subjects.txt')
        with open(subjects_txt_path, 'w') as f:
            for subject_id in self.common_subjects:
                f.write(f"{subject_id}\n")
        log.info(f"Saved {len(self.common_subjects)} common subjects to {subjects_txt_path}")

    def print_summary(self) -> None:
        """Print a summary of the CKA coherence analysis."""
        if self.cka_matrix is None:
            print("No analysis results available. Run analysis first.")
            return

        stats = self.get_coherence_stats()

        print("\n" + "="*70)
        print("CKA COHERENCE ANALYSIS SUMMARY")
        print("="*70)
        print(f"\nNumber of models: {stats['n_models']}")
        print(f"Number of common subjects: {len(self.common_subjects)}")
        print(f"Number of model pairs: {stats['n_pairs']}")

        print(f"\nCoherence Statistics:")
        print(f"  Mean CKA:   {stats['mean_cka']:.4f}")
        print(f"  Median CKA: {stats['median_cka']:.4f}")
        print(f"  Std CKA:    {stats['std_cka']:.4f}")
        print(f"  Min CKA:    {stats['min_cka']:.4f}")
        print(f"  Max CKA:    {stats['max_cka']:.4f}")

        print(f"\nCKA Matrix:")
        print(self.cka_matrix.to_string())
        print("="*70 + "\n")


def test_models_coherence_from_directory(
    models_dir: str,
    embedding_filename: str = 'full_embeddings.csv',
    output_dir: Optional[str] = None,
    subject_column: str = 'ID',
    embeddings_key: str = 'embeddings',
    subject_key: str = 'subject_ids',
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Test coherence of all models in a directory.

    This is a convenience function that automatically discovers models
    in a directory structure and tests their coherence using recursive search.
    Both CSV (``.csv``) and PyTorch (``.pt``) embedding files are supported.

    Parameters
    ----------
    models_dir : str
        Directory containing model subdirectories.
        Files are searched recursively within this directory.
    embedding_filename : str, optional
        Name of the embedding file to look for. Default is 'full_embeddings.csv'.
        Use e.g. ``'embeddings.pt'`` to search for PyTorch files instead.
    output_dir : str, optional
        Directory to save results. If None, uses models_dir/cka_coherence.
    subject_column : str, optional
        Column name for subject IDs in CSV files. Default is 'ID'.
    embeddings_key : str, optional
        Dict key for the embedding tensor in PT files. Default is 'embeddings'.
    subject_key : str, optional
        Dict key for subject IDs in PT files. Default is 'subject_ids'.

    Returns
    -------
    cka_matrix : pd.DataFrame
        Pairwise CKA similarity matrix.
    stats : dict
        Summary statistics.

    Examples
    --------
    >>> # CSV (default)
    >>> cka_matrix, stats = test_models_coherence_from_directory(
    ...     'data/TEST01/derivatives/champollion_V1/models',
    ... )
    >>> # PyTorch .pt files
    >>> cka_matrix, stats = test_models_coherence_from_directory(
    ...     'data/TEST01/derivatives/champollion_V1/models',
    ...     embedding_filename='embeddings.pt',
    ... )
    >>> print(f"Mean coherence: {stats['mean_cka']:.4f}")
    """
    models_dir_path = Path(models_dir)

    if not models_dir_path.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    # Recursively search for embedding files
    embedding_paths = {}
    found_files = list(models_dir_path.rglob(embedding_filename))

    if not found_files:
        raise ValueError(
            f"No embedding files named '{embedding_filename}' found "
            f"recursively in {models_dir}"
        )

    for embedding_file in found_files:
        model_name = embedding_file.parent.name

        # Handle duplicate model names by using more path context
        if model_name in embedding_paths:
            relative_path = embedding_file.relative_to(models_dir_path)
            model_name = str(relative_path.parent).replace('/', '_')

        embedding_paths[model_name] = str(embedding_file)
        log.info(f"Found embeddings for model: {model_name} at {embedding_file}")

    log.info(f"Found {len(embedding_paths)} models with embeddings (recursive search)")

    # Run analysis
    tester = CKACoherenceTester(
        embedding_paths,
        subject_column=subject_column,
        embeddings_key=embeddings_key,
        subject_key=subject_key,
    )
    cka_matrix, stats = tester.run_full_analysis()

    # Save results
    if output_dir is None:
        output_dir = str(models_dir_path / 'cka_coherence')

    os.makedirs(output_dir, exist_ok=True)
    tester.save_results(str(output_dir))
    tester.print_summary()

    return cka_matrix, stats


def load_state_dict_pt(pt_path: str) -> Dict[str, np.ndarray]:
    """
    Load a model checkpoint and return its weight tensors as numpy arrays.

    Only 2-D-compatible parameters (weight matrices and conv kernels) are
    kept; 1-D parameters (biases, batch-norm scales) are skipped because
    CKA requires at least 2 rows.

    Parameters
    ----------
    pt_path : str
        Path to a ``.pt`` checkpoint containing either a raw ``state_dict``
        or a dict with a ``'state_dict'`` key.

    Returns
    -------
    weights : dict
        Mapping from layer name to numpy array of shape (n_rows, n_cols).
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required to load .pt files: pip install torch") from exc

    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Checkpoint not found: {pt_path}")

    data = torch.load(pt_path, map_location='cpu', weights_only=False)

    state_dict = data.get('state_dict', data) if isinstance(data, dict) else data
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Expected a dict or a dict with 'state_dict', got {type(data)} in {pt_path}"
        )

    weights = {}
    for name, tensor in state_dict.items():
        arr = tensor.detach().float().numpy()
        if arr.ndim == 1:
            continue  # skip biases / 1-D params
        # Reshape conv kernels (out, in, *spatial) → (out, in*spatial)
        weights[name] = arr.reshape(arr.shape[0], -1)

    log.info(f"Loaded {len(weights)} weight matrices from {pt_path}")
    return weights


def compute_weight_cka(
    path_a: str,
    name_a: str,
    path_b: str,
    name_b: str,
    output_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Compare two model checkpoints layer-by-layer using linear CKA.

    Parameters
    ----------
    path_a, path_b : str
        Paths to ``.pt`` checkpoint files.
    name_a, name_b : str
        Display names for the two models.
    output_dir : str, optional
        Directory to save per-layer results and stats JSON.
        Defaults to ``./cka_coherence_weights``.

    Returns
    -------
    results : pd.DataFrame
        Per-layer CKA scores with columns ``['layer', 'cka']``.
    stats : dict
        Summary statistics (mean, median, std, min, max).
    """
    weights_a = load_state_dict_pt(path_a)
    weights_b = load_state_dict_pt(path_b)

    common_layers = [l for l in weights_a if l in weights_b]
    skipped = []

    if not common_layers:
        raise ValueError(
            f"No common layer names found between {name_a} and {name_b}. "
            f"Layers in {name_a}: {list(weights_a.keys())[:5]} …"
        )

    rows = []
    for layer in common_layers:
        W1, W2 = weights_a[layer], weights_b[layer]
        if W1.shape != W2.shape:
            skipped.append((layer, W1.shape, W2.shape))
            continue
        if W1.shape[0] < 2:
            skipped.append((layer, W1.shape, W2.shape))
            continue
        cka = linear_CKA(W1, W2)
        rows.append({'layer': layer, 'cka': cka})
        log.debug(f"CKA({layer}) = {cka:.4f}")

    if skipped:
        log.warning(
            f"Skipped {len(skipped)} layers (shape mismatch or too few rows): "
            + ", ".join(l for l, *_ in skipped[:5])
            + (" …" if len(skipped) > 5 else "")
        )

    if not rows:
        raise ValueError("No valid common layers found to compute CKA.")

    results = pd.DataFrame(rows)
    cka_values = results['cka'].values
    stats = {
        'mean_cka': float(np.mean(cka_values)),
        'median_cka': float(np.median(cka_values)),
        'std_cka': float(np.std(cka_values)),
        'min_cka': float(np.min(cka_values)),
        'max_cka': float(np.max(cka_values)),
        'n_layers': len(rows),
        'n_skipped': len(skipped),
    }

    # Print summary
    print("\n" + "=" * 70)
    print(f"WEIGHT CKA: {name_a}  vs  {name_b}")
    print("=" * 70)
    print(f"\nLayers compared : {stats['n_layers']}  (skipped: {stats['n_skipped']})")
    print(f"\nPer-layer CKA scores:")
    for _, row in results.iterrows():
        print(f"  {row['layer']:<60s}  {row['cka']:.4f}")
    print(f"\nSummary:")
    print(f"  Mean CKA:   {stats['mean_cka']:.4f}")
    print(f"  Median CKA: {stats['median_cka']:.4f}")
    print(f"  Std CKA:    {stats['std_cka']:.4f}")
    print(f"  Min CKA:    {stats['min_cka']:.4f}  ({results.loc[results['cka'].idxmin(), 'layer']})")
    print(f"  Max CKA:    {stats['max_cka']:.4f}  ({results.loc[results['cka'].idxmax(), 'layer']})")
    print("=" * 70 + "\n")

    # Save
    if output_dir is None:
        output_dir = './cka_coherence_weights'
    os.makedirs(output_dir, exist_ok=True)
    results.to_csv(os.path.join(output_dir, 'weight_cka_per_layer.csv'), index=False)
    with open(os.path.join(output_dir, 'weight_cka_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    log.info(f"Results saved to {output_dir}")

    return results, stats


def _build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute pairwise CKA coherence between model embeddings or weights.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Embedding files (CSV or PT)
  python -m contrastive.evaluation.cka_coherence \\
      model_a:path/to/a.pt model_b:path/to/b.pt

  # Discover all embedding files under a directory
  python -m contrastive.evaluation.cka_coherence \\
      --dir data/models --filename embeddings.pt

  # Weight (state_dict) comparison between two checkpoints
  python -m contrastive.evaluation.cka_coherence --weights \\
      model_a:path/to/best_model_weights.pt model_b:path/to/best_model_weights.pt
""",
    )

    parser.add_argument(
        'files',
        nargs='*',
        metavar='NAME:PATH',
        help="Embedding files as name:path pairs (CSV or PT). "
             "Mutually exclusive with --dir.",
    )
    parser.add_argument(
        '--dir',
        metavar='DIR',
        help="Directory to search recursively for embedding files.",
    )
    parser.add_argument(
        '--filename',
        default='full_embeddings.csv',
        metavar='FILENAME',
        help="Filename to search for when using --dir (default: full_embeddings.csv).",
    )
    parser.add_argument(
        '--output-dir', '-o',
        metavar='DIR',
        help="Directory to save results (default: <dir>/cka_coherence or ./cka_coherence).",
    )
    parser.add_argument(
        '--subject-column',
        default='ID',
        metavar='COL',
        help="Subject ID column name for CSV files (default: ID).",
    )
    parser.add_argument(
        '--embeddings-key',
        default='embeddings',
        metavar='KEY',
        help="Dict key for embeddings in PT files (default: embeddings).",
    )
    parser.add_argument(
        '--subject-key',
        default='subject_ids',
        metavar='KEY',
        help="Dict key for subject IDs in PT files (default: subject_ids).",
    )
    parser.add_argument(
        '--weights',
        action='store_true',
        help="Compare model checkpoints (state_dict) layer-by-layer instead of embeddings.",
    )
    return parser


if __name__ == '__main__':
    parser = _build_parser()
    args = parser.parse_args()

    if args.weights:
        # --- weight CKA mode ---
        if args.dir:
            parser.error("--dir is not supported with --weights; pass two NAME:PATH files.")
        if len(args.files) != 2:
            parser.error("--weights requires exactly two NAME:PATH arguments.")
        tokens = []
        for token in args.files:
            if ':' not in token:
                parser.error(f"Expected NAME:PATH, got '{token}'")
            name, path = token.split(':', 1)
            tokens.append((name, path))
        (name_a, path_a), (name_b, path_b) = tokens
        compute_weight_cka(
            path_a, name_a,
            path_b, name_b,
            output_dir=args.output_dir,
        )
    else:
        # --- embedding CKA mode ---
        if args.dir and args.files:
            parser.error("Specify either positional NAME:PATH files or --dir, not both.")
        if not args.dir and not args.files:
            parser.error("Provide at least one NAME:PATH file or use --dir.")

        if args.dir:
            test_models_coherence_from_directory(
                args.dir,
                embedding_filename=args.filename,
                output_dir=args.output_dir,
                subject_column=args.subject_column,
                embeddings_key=args.embeddings_key,
                subject_key=args.subject_key,
            )
        else:
            embedding_paths = {}
            for token in args.files:
                if ':' not in token:
                    parser.error(f"Expected NAME:PATH, got '{token}'")
                name, path = token.split(':', 1)
                embedding_paths[name] = path

            output_dir = args.output_dir or './cka_coherence'
            tester = CKACoherenceTester(
                embedding_paths,
                subject_column=args.subject_column,
                embeddings_key=args.embeddings_key,
                subject_key=args.subject_key,
            )
            tester.run_full_analysis()
            tester.save_results(output_dir)
            tester.print_summary()
