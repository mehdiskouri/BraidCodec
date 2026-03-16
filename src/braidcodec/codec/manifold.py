"""Deterministic hypergraph manifold construction and compact state fitting.

Phase D utilities:
- build a canonical hypergraph from frequency tokens
- fit compact low-dimensional manifold state from oscillator points
- emit stable hashes for reproducibility and integrity checks
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import blake3
import numpy as np

if TYPE_CHECKING:
    from braidcodec.codec.oscillator_normalization import OscillatorPoint
    from braidcodec.codec.tokenizer_frequency import FrequencyToken

MANIFOLD_PROFILE_ID = "hypergraph-manifold-v1"


@dataclass(frozen=True, slots=True)
class HyperNode:
    """Canonical token node in hypergraph space."""

    node_id: int
    token_index: int
    layer_index: int
    bin_index: int


@dataclass(frozen=True, slots=True)
class HyperEdge:
    """Directed edge between consecutive token nodes."""

    src: int
    dst: int
    weight_fp: int


@dataclass(frozen=True, slots=True)
class HyperLayerSummary:
    """Deterministic aggregate for one layer."""

    layer_index: int
    node_count: int
    unique_bin_count: int
    commitment_hash64: int


@dataclass(frozen=True, slots=True)
class HypergraphModel:
    """Compact hypergraph model derived from tokens."""

    layer_count: int
    nodes: tuple[HyperNode, ...]
    edges: tuple[HyperEdge, ...]
    layers: tuple[HyperLayerSummary, ...]
    graph_hash: str


@dataclass(frozen=True, slots=True)
class ManifoldState:
    """Low-dimensional deterministic manifold state."""

    profile_id: str
    layer_count: int
    layer_vectors: tuple[tuple[float, ...], ...]
    global_vector: tuple[float, ...]
    state_hash: str


@dataclass(frozen=True, slots=True)
class CouplingMatrixSummary:
    """Deterministic summary over token coupling matrix statistics."""

    dimension: int
    nnz: int
    density: float
    spectral_radius: float
    matrix_hash: str


def _layer_index(token: FrequencyToken, *, layer_count: int) -> int:
    payload = (
        f"{token.token_class}|{token.phoneme_tag or '-'}|{token.bin_index}|{token.value}"
    ).encode()
    digest = blake3.blake3(payload).digest(length=8)
    return int.from_bytes(digest, byteorder="big", signed=False) % max(layer_count, 1)


def build_deterministic_hypergraph(
    tokens: tuple[FrequencyToken, ...],
    *,
    layer_count: int = 8,
) -> HypergraphModel:
    """Build canonical hypergraph from tokenizer output."""
    if layer_count < 1:
        raise ValueError("layer_count must be >= 1")

    nodes: list[HyperNode] = []
    for i, token in enumerate(tokens):
        layer = _layer_index(token, layer_count=layer_count)
        nodes.append(
            HyperNode(
                node_id=i,
                token_index=i,
                layer_index=layer,
                bin_index=token.bin_index,
            )
        )

    edges: list[HyperEdge] = []
    for i in range(max(0, len(nodes) - 1)):
        a = nodes[i]
        b = nodes[i + 1]
        # Fixed-point integer weight for deterministic serialization.
        weight_fp = 2000 if a.layer_index == b.layer_index else 1000
        edges.append(HyperEdge(src=a.node_id, dst=b.node_id, weight_fp=weight_fp))

    layer_summaries: list[HyperLayerSummary] = []
    for layer_idx in range(layer_count):
        layer_nodes = [n for n in nodes if n.layer_index == layer_idx]
        bins = sorted({n.bin_index for n in layer_nodes})
        payload = json.dumps(
            {
                "layer": layer_idx,
                "nodes": [n.node_id for n in layer_nodes],
                "bins": bins,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest64 = int.from_bytes(blake3.blake3(payload).digest(length=8), byteorder="big")
        layer_summaries.append(
            HyperLayerSummary(
                layer_index=layer_idx,
                node_count=len(layer_nodes),
                unique_bin_count=len(bins),
                commitment_hash64=digest64,
            )
        )

    graph_payload = json.dumps(
        {
            "layer_count": layer_count,
            "nodes": [(n.node_id, n.layer_index, n.bin_index) for n in nodes],
            "edges": [(e.src, e.dst, e.weight_fp) for e in edges],
            "layers": [
                (
                    layer.layer_index,
                    layer.node_count,
                    layer.unique_bin_count,
                    layer.commitment_hash64,
                )
                for layer in layer_summaries
            ],
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    graph_hash = blake3.blake3(graph_payload).hexdigest()

    return HypergraphModel(
        layer_count=layer_count,
        nodes=tuple(nodes),
        edges=tuple(edges),
        layers=tuple(layer_summaries),
        graph_hash=graph_hash,
    )


def fit_compact_manifold_state(
    points: tuple[OscillatorPoint, ...],
    graph: HypergraphModel,
) -> ManifoldState:
    """Fit deterministic low-dimensional manifold vectors from graph+points."""
    if len(points) != len(graph.nodes):
        raise ValueError("points count must match graph node count")

    by_layer: dict[int, list[int]] = {layer.layer_index: [] for layer in graph.layers}
    for idx, node in enumerate(graph.nodes):
        by_layer[node.layer_index].append(idx)

    layer_vectors: list[tuple[float, ...]] = []
    for layer in graph.layers:
        idxs = by_layer[layer.layer_index]
        if not idxs:
            layer_vectors.append((0.0, 0.0, 0.0, 0.0, 0.0))
            continue

        omega = np.asarray([points[i].omega for i in idxs], dtype=np.float64)
        amp = np.asarray([points[i].amplitude for i in idxs], dtype=np.float64)
        phase = np.asarray([points[i].phase for i in idxs], dtype=np.float64)
        scale = np.asarray([points[i].scale for i in idxs], dtype=np.float64)

        layer_vec = (
            float(np.mean(omega)),
            float(np.mean(amp)),
            float(np.mean(phase)),
            float(np.mean(scale)),
            float(layer.node_count / max(len(graph.nodes), 1)),
        )
        layer_vectors.append(layer_vec)

    # Global 5D state vector.
    all_omega = np.asarray([p.omega for p in points], dtype=np.float64)
    all_amp = np.asarray([p.amplitude for p in points], dtype=np.float64)
    all_phase = np.asarray([p.phase for p in points], dtype=np.float64)
    all_scale = np.asarray([p.scale for p in points], dtype=np.float64)

    global_vec = (
        float(np.mean(all_omega)),
        float(np.mean(all_amp)),
        float(np.mean(all_phase)),
        float(np.mean(all_scale)),
        float(len(graph.edges) / max(len(graph.nodes), 1)),
    )

    state_payload = json.dumps(
        {
            "profile_id": MANIFOLD_PROFILE_ID,
            "layer_count": graph.layer_count,
            "graph_hash": graph.graph_hash,
            "layer_vectors": layer_vectors,
            "global_vector": global_vec,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    state_hash = blake3.blake3(state_payload).hexdigest()

    return ManifoldState(
        profile_id=MANIFOLD_PROFILE_ID,
        layer_count=graph.layer_count,
        layer_vectors=tuple(layer_vectors),
        global_vector=global_vec,
        state_hash=state_hash,
    )


def manifold_seed_vector(state: ManifoldState) -> np.ndarray:
    """Produce deterministic seed vector for fixed-point initialization."""
    flat: list[float] = []
    for vec in state.layer_vectors:
        flat.extend(vec)
    flat.extend(state.global_vector)
    return np.asarray(flat, dtype=np.float64)


def manifold_metadata(state: ManifoldState) -> dict[str, str]:
    """Return stable metadata fields for reconstructive streams."""
    return {
        "manifold_profile_id": state.profile_id,
        "manifold_layer_count": str(state.layer_count),
        "manifold_state_hash": state.state_hash,
        "manifold_latent_dim": str(len(state.global_vector)),
    }


def build_coupling_matrix_summary(
    tokens: tuple[FrequencyToken, ...],
    *,
    window: int = 4,
    matrix_dim: int = 128,
) -> CouplingMatrixSummary:
    """Build deterministic token coupling matrix summary.

    Matrix cells count local co-occurrence of token frequency bins within a
    bounded context window. This provides NNZ and spectral diagnostics that can
    be reused by compact reconstructive codec selection.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    if matrix_dim < 2:
        raise ValueError("matrix_dim must be >= 2")

    mat = np.zeros((matrix_dim, matrix_dim), dtype=np.float64)
    n = len(tokens)
    for i in range(n):
        src = int(tokens[i].bin_index) % matrix_dim
        j_end = min(i + window + 1, n)
        for j in range(i + 1, j_end):
            dst = int(tokens[j].bin_index) % matrix_dim
            w = 1.0 / float(j - i)
            mat[src, dst] += w
            mat[dst, src] += w

    nnz = int(np.count_nonzero(mat))
    density = float(nnz / float(matrix_dim * matrix_dim))

    if nnz == 0:
        spectral_radius = 0.0
    else:
        vec = np.full(matrix_dim, 1.0 / np.sqrt(float(matrix_dim)), dtype=np.float64)
        for _ in range(12):
            nxt = mat @ vec
            norm = float(np.linalg.norm(nxt))
            if norm <= 1e-12:
                vec = nxt
                break
            vec = nxt / norm
        spectral_radius = float(np.linalg.norm(mat @ vec))

    matrix_hash = blake3.blake3(mat.tobytes(order="C")).hexdigest()
    return CouplingMatrixSummary(
        dimension=matrix_dim,
        nnz=nnz,
        density=density,
        spectral_radius=spectral_radius,
        matrix_hash=matrix_hash,
    )


def coupling_matrix_metadata(summary: CouplingMatrixSummary) -> dict[str, str]:
    """Serialize coupling matrix summary into reconstructive metadata fields."""
    return {
        "coupling_matrix_dim": str(summary.dimension),
        "coupling_matrix_nnz": str(summary.nnz),
        "coupling_matrix_density": f"{summary.density:.17g}",
        "coupling_matrix_spectral_radius": f"{summary.spectral_radius:.17g}",
        "coupling_matrix_hash": summary.matrix_hash,
    }
