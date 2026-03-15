"""Encoder — full pipeline from bytes to ``EncodedStream``.

Maps raw data through chunking → braid construction → invariant computation
→ optional simplification → ``EncodedStream`` assembly.  Parallel block
encoding via ``ProcessPoolExecutor`` for multi-block payloads.
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import TYPE_CHECKING

import blake3
import numpy as np

from braidcodec.algebra.braid_equations import (
    BraidEquation,
    contract_braid_tensor,
    jones_polynomial,
    writhe,
)
from braidcodec.algebra.fixedpoint import FixedPointResult, iterate_fixedpoint
from braidcodec.algebra.tsr_constants import ETA, FIXEDPOINT_MAX_ITER, FIXEDPOINT_TOL, KAPPA
from braidcodec.codec.chunker import (
    bytes_to_generators,
    chunk_stream,
    compute_block_size,
)
from braidcodec.codec.manifold import (
    CouplingMatrixSummary,
    ManifoldState,
    build_coupling_matrix_summary,
    build_deterministic_hypergraph,
    coupling_matrix_metadata,
    fit_compact_manifold_state,
    manifold_metadata,
    manifold_seed_vector,
)
from braidcodec.codec.oscillator_normalization import OscillatorNormalizationV1
from braidcodec.codec.preprocessing import (
    build_chunk_profiles,
    mean_estimated_cost,
    morton_key_1d,
    sort_chunk_indices_for_topology,
    synthesize_topology_generators_v2,
    topology_commitment_v2,
)
from braidcodec.codec.reconstructive_compact import fit_reconstructive_program
from braidcodec.codec.reconstructive_transform import reconstructive_forward_generators
from braidcodec.codec.schema import (
    EncodedBlock,
    EncodedStream,
    build_reconstructive_payload_metadata,
    compute_reconstructive_commitment,
    validate_reconstructive_metadata,
)
from braidcodec.codec.tokenizer_frequency import (
    DomainKind,
    FrequencyTokenizerV1,
    TokenizationResult,
)

if TYPE_CHECKING:
    from braidcodec.crypto.keys import BraidKey


# ── Tier threshold ────────────────────────────────────────────────────────

_TIER_2_MAX_GENERATORS: int = 12
_VALID_PREPROCESSING_MODES: frozenset[str] = frozenset({"topology", "legacy", "reconstructive"})
_RECONSTRUCTIVE_PAYLOAD_KEY = "reconstructive_payload_v1"
_RECONSTRUCTIVE_PAYLOAD_KEY_SHORT = "rp1"

_BlockTask = tuple[
    list[int],
    int,
    str,
    dict[str, float] | None,
    int,
    int,
    list[int] | None,
    int | None,
    int | None,
    int | None,
    float | None,
    int | None,
    int | None,
    int | None,
    int | None,
    int | None,
    int | None,
]


# ── Module-level block encoder (picklable for ProcessPoolExecutor) ────────


def _encode_block(
    generators: list[int],
    n_strands: int,
    sector: str,
    sector_params: dict[str, float] | None,
    block_index: int,
    original_length: int,
    decode_generators: list[int] | None = None,
    topology_layer_index: int | None = None,
    topology_layer_n_chunks: int | None = None,
    topology_nnz_bits: int | None = None,
    topology_dt_scale: float | None = None,
    topology_hash32: int | None = None,
    topology_density_fp: int | None = None,
    topology_centroid_fp: int | None = None,
    topology_variance_fp: int | None = None,
    topology_morton_key: int | None = None,
    topology_commitment_value: int | None = None,
) -> EncodedBlock:
    """Encode a single chunk into an ``EncodedBlock``.

    This function is module-level (not a closure) so it can be pickled by
    ``ProcessPoolExecutor``.
    """
    braid = BraidEquation(
        n_strands,
        generators,
        sector=sector,
        _sector_params=sector_params,
    )

    w = writhe(braid)

    # Tier selection: Jones (expensive state-sum) for short braids only.
    tier = 2 if len(generators) <= _TIER_2_MAX_GENERATORS else 3

    jones_real: float | None = None
    jones_imag: float | None = None
    trace_real: float | None = None
    trace_imag: float | None = None

    if tier == 2:
        j = jones_polynomial(braid)
        jones_real = j.real
        jones_imag = j.imag
    else:
        matrix = contract_braid_tensor(braid)
        tr = complex(np.trace(matrix))
        trace_real = tr.real
        trace_imag = tr.imag

    return EncodedBlock(
        generators=generators,
        n_strands=n_strands,
        sector=sector,
        writhe=w,
        block_index=block_index,
        original_length=original_length,
        invariant_tier=tier,
        jones_real=jones_real,
        jones_imag=jones_imag,
        trace_real=trace_real,
        trace_imag=trace_imag,
        decode_generators=decode_generators,
        topology_layer_index=topology_layer_index,
        topology_layer_n_chunks=topology_layer_n_chunks,
        topology_nnz_bits=topology_nnz_bits,
        topology_dt_scale=topology_dt_scale,
        topology_hash32=topology_hash32,
        topology_density_fp=topology_density_fp,
        topology_centroid_fp=topology_centroid_fp,
        topology_variance_fp=topology_variance_fp,
        topology_morton_key=topology_morton_key,
        topology_commitment=topology_commitment_value,
    )


def _encode_batch(batch: list[_BlockTask]) -> list[EncodedBlock]:
    """Encode a batch of blocks in one worker call.

    Batch-level execution reduces scheduler/IPC overhead versus one future
    per block, especially for large 1MB+ workloads.
    """
    return [_encode_block(*task) for task in batch]


def _get_reconstructive_payload_blob(metadata: dict[str, str]) -> str:
    """Return reconstructive payload blob from short or legacy key."""
    payload = metadata.get(_RECONSTRUCTIVE_PAYLOAD_KEY_SHORT, "")
    if payload:
        return payload
    return metadata.get(_RECONSTRUCTIVE_PAYLOAD_KEY, "")


# ── Public API ────────────────────────────────────────────────────────────


def encode(
    data: bytes,
    key: BraidKey,
    *,
    generators_per_block: int = 32,
    max_workers: int | None = None,
    preprocessing_mode: str = "topology",
    reconstructive_domain: str | None = None,
    reconstructive_tokenization: TokenizationResult | None = None,
) -> EncodedStream:
    """Encode raw data into an ``EncodedStream``.

    Parameters
    ----------
    data:
        Raw bytes to encode.
    key:
        A ``BraidKey`` controlling the R-matrix parametrisation.
    generators_per_block:
        Number of generators per block (default 32).
    max_workers:
        Maximum number of parallel workers.  ``None`` uses
        ``min(cpu_count()-1, 8)``.  Set to 1 for serial execution.
    preprocessing_mode:
        Chunk preprocessing mode. ``"topology"`` (default) computes deterministic
        BFPS/topology profiles and applies regime-aware scheduling. ``"legacy"``
        bypasses preprocessing and uses historical chunk order.
    reconstructive_domain:
        Optional domain override for reconstructive preprocessing mode.
        Allowed values: ``"text"``, ``"json"``, ``"logs"``.
    reconstructive_tokenization:
        Optional precomputed tokenizer result for reconstructive mode. When
        provided, encoder skips tokenizer pass and uses this deterministic
        token stream directly for normalization/manifold/fixed-point stages.

    Returns
    -------
    EncodedStream
        Complete encoded payload ready for wire serialization.
    """
    if preprocessing_mode not in _VALID_PREPROCESSING_MODES:
        valid = ", ".join(sorted(_VALID_PREPROCESSING_MODES))
        raise ValueError(f"preprocessing_mode must be one of: {valid}")

    t_start = time.perf_counter()
    block_size = compute_block_size(key.n_strands, generators_per_block)
    checksum = blake3.blake3(data).digest()
    chunks = list(chunk_stream(data, block_size))

    sector_params = key.sector_params

    chunk_by_index = {idx: (chunk, orig_len) for idx, chunk, orig_len in chunks}
    profile_elapsed = 0.0
    layer_order_elapsed = 0.0
    reconstructive_meta: dict[str, str] = {}
    reconstructive_seed_vector: str | None = None

    if preprocessing_mode == "reconstructive":
        reconstructive_meta = _build_reconstructive_metadata(
            data,
            domain_override=reconstructive_domain,
            tokenization_override=reconstructive_tokenization,
        )
        reconstructive_seed_vector = reconstructive_meta.get("km_seed_vector")

        metadata = {
            "preprocessing_mode": preprocessing_mode,
            "execution_mode": "reconstructive-compact",
            "task_count": "0",
            "batch_count": "0",
            "batch_size": "0",
            "estimated_cost_mean": "0.000000",
            "timing_preprocess_s": "0.000000",
            "timing_layer_order_s": "0.000000",
            "timing_encode_core_s": "0.000000",
            "timing_total_s": f"{(time.perf_counter() - t_start):.6f}",
            **reconstructive_meta,
        }
        payload_json = _get_reconstructive_payload_blob(metadata)
        if isinstance(payload_json, str) and payload_json:
            metadata["reconstructive_commitment"] = compute_reconstructive_commitment(
                payload_json,
                tuple(),
            )

        return EncodedStream(
            blocks=tuple(),
            n_strands=key.n_strands,
            sector=key.sector,
            total_bytes=len(data),
            checksum=checksum,
            metadata=metadata,
        )

    if preprocessing_mode == "topology":
        t0 = time.perf_counter()
        profiles = build_chunk_profiles(
            chunks,
            n_strands=key.n_strands,
            generators_per_block=generators_per_block,
        )
        profile_elapsed = time.perf_counter() - t0
        t0 = time.perf_counter()
        chunk_order = sort_chunk_indices_for_topology(profiles)
        layer_order_elapsed = time.perf_counter() - t0
        estimated_cost = mean_estimated_cost(profiles)
        profile_by_index = {p.block_index: p for p in profiles}
    else:
        chunk_order = [idx for idx, _, _ in chunks]
        estimated_cost = 0.0
        profile_by_index = {}

    # Build per-block tasks for _encode_block.
    tasks_with_layer: list[tuple[int, _BlockTask]] = []
    for idx in chunk_order:
        profile = profile_by_index.get(idx)
        chunk = chunk_by_index[idx][0]
        base_generators = bytes_to_generators(chunk, key.n_strands, generators_per_block)

        if profile is not None:
            morton_key = morton_key_1d(idx, profile.layer_index)
            block_generators = synthesize_topology_generators_v2(
                generators=base_generators,
                n_strands=key.n_strands,
                layer_index=profile.layer_index,
                signature_hash32=profile.signature.hash32,
                morton_key=morton_key,
                nnz_bits=profile.nnz_bits,
            )
            decode_generators = None
            topo_commitment = topology_commitment_v2(
                morton_key=morton_key,
                layer_index=profile.layer_index,
                hash32=profile.signature.hash32,
            )
        else:
            if preprocessing_mode == "reconstructive":
                if not reconstructive_seed_vector:
                    raise ValueError("reconstructive metadata missing km_seed_vector")
                block_generators = reconstructive_forward_generators(
                    base_generators,
                    n_strands=key.n_strands,
                    seed_vector=reconstructive_seed_vector,
                    block_index=idx,
                )
            else:
                block_generators = base_generators
            decode_generators = None
            morton_key = None
            topo_commitment = None

        tasks_with_layer.append(
            (
                profile.layer_index if profile is not None else -1,
                (
                    block_generators,
                    key.n_strands,
                    key.sector,
                    sector_params,
                    idx,
                    chunk_by_index[idx][1],
                    decode_generators,
                    profile.layer_index if profile is not None else None,
                    None,
                    profile.nnz_bits if profile is not None else None,
                    None,
                    profile.signature.hash32 if profile is not None else None,
                    None,
                    None,
                    None,
                    morton_key,
                    topo_commitment,
                ),
            )
        )

    # Determine effective worker count.
    if max_workers is None:
        effective_workers = min(max((os.cpu_count() or 1) - 1, 1), 8, len(tasks_with_layer))
    else:
        effective_workers = min(max(max_workers, 1), 8, len(tasks_with_layer))

    exec_mode = _select_execution_mode(
        task_count=len(tasks_with_layer),
        effective_workers=effective_workers,
        estimated_cost=estimated_cost,
        n_strands=key.n_strands,
        generators_per_block=generators_per_block,
    )

    batch_size = _choose_batch_size(
        task_count=len(tasks_with_layer),
        effective_workers=effective_workers,
        estimated_cost=estimated_cost,
        exec_mode=exec_mode,
    )
    batches = _build_layer_aware_batches(tasks_with_layer, batch_size=batch_size)

    t0 = time.perf_counter()
    if exec_mode == "serial":
        blocks = [block for batch in batches for block in _encode_batch(batch)]
    elif exec_mode == "thread":
        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            futures = [pool.submit(_encode_batch, batch) for batch in batches]
            blocks = [block for fut in futures for block in fut.result()]
    else:
        with ProcessPoolExecutor(max_workers=effective_workers) as pool:
            futures = [pool.submit(_encode_batch, batch) for batch in batches]
            blocks = [block for fut in futures for block in fut.result()]
    encode_elapsed = time.perf_counter() - t0

    blocks.sort(key=lambda b: b.block_index)
    total_elapsed = time.perf_counter() - t_start

    metadata = {
        "preprocessing_mode": preprocessing_mode,
        "execution_mode": exec_mode,
        "task_count": str(len(tasks_with_layer)),
        "batch_count": str(len(batches)),
        "batch_size": str(batch_size),
        "estimated_cost_mean": f"{estimated_cost:.6f}",
        "timing_preprocess_s": f"{profile_elapsed:.6f}",
        "timing_layer_order_s": f"{layer_order_elapsed:.6f}",
        "timing_encode_core_s": f"{encode_elapsed:.6f}",
        "timing_total_s": f"{total_elapsed:.6f}",
        **(
            {
                "topology_synthesis_version": "2",
                "topology_decode_strategy": "inverse-transform-v1",
            }
            if preprocessing_mode == "topology"
            else {}
        ),
        **reconstructive_meta,
    }

    if preprocessing_mode == "reconstructive":
        payload_json = _get_reconstructive_payload_blob(metadata)
        if isinstance(payload_json, str) and payload_json:
            metadata["reconstructive_commitment"] = compute_reconstructive_commitment(
                payload_json,
                tuple(blocks),
            )

    return EncodedStream(
        blocks=tuple(blocks),
        n_strands=key.n_strands,
        sector=key.sector,
        total_bytes=len(data),
        checksum=checksum,
        metadata=metadata,
    )


def _build_reconstructive_fidelity_bundle(
    *,
    state: ManifoldState,
    fixedpoint: FixedPointResult,
    coupling: CouplingMatrixSummary,
) -> dict[str, str]:
    """Build lightweight fidelity object metrics for reconstructive streams.

    These are deterministic scalar proxies that expose energy/topology/coherence
    style diagnostics in metadata without changing decode authority gates.
    """
    layer_vectors = state.layer_vectors
    global_vec = state.global_vector
    diag = fixedpoint.diagnostics

    topology_proxy = float(coupling.nnz) if coupling.nnz > 0 else 0.0

    energy_proxy = 0.0
    if layer_vectors:
        weighted: list[float] = []
        for vec in layer_vectors:
            if len(vec) >= 2:
                weighted.append(float(vec[0]) * float(vec[1]))
        if weighted:
            energy_proxy = float(np.mean(weighted))
    elif len(global_vec) >= 2:
        energy_proxy = float(global_vec[0]) * float(global_vec[1])

    coherence_proxy = float(getattr(diag, "converged_ratio", 0.0)) * (
        1.0 / (1.0 + float(getattr(diag, "residual_mean", 0.0)) * 1000.0)
    )

    payload = {
        "energy_proxy": f"{energy_proxy:.17g}",
        "topology_proxy": f"{topology_proxy:.17g}",
        "coherence_proxy": f"{coherence_proxy:.17g}",
    }
    return {
        "fidelity_energy": payload["energy_proxy"],
        "fidelity_topology": payload["topology_proxy"],
        "fidelity_coherence": payload["coherence_proxy"],
        "fidelity_bundle_v1": json.dumps(payload, sort_keys=True, separators=(",", ":")),
    }


def _infer_reconstructive_domain(data: bytes) -> DomainKind:
    """Infer tokenizer domain for reconstructive mode deterministically."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "reconstructive mode currently supports UTF-8 Text/JSON/Logs only"
        ) from exc

    try:
        json.loads(text)
    except Exception:
        pass
    else:
        return "json"

    if re.search(r"\d{4}-\d{2}-\d{2}[T ][0-9:.+-Z]*", text):
        return "logs"
    return "text"


def _build_reconstructive_metadata(
    data: bytes,
    *,
    domain_override: str | None,
    tokenization_override: TokenizationResult | None = None,
) -> dict[str, str]:
    """Compute deterministic reconstructive metadata contract payload."""
    tokenizer = FrequencyTokenizerV1()
    normalizer = OscillatorNormalizationV1()

    if tokenization_override is not None:
        domain = tokenization_override.domain
        if domain_override is not None and domain_override.strip().lower() != domain:
            raise ValueError(
                "reconstructive_domain does not match provided reconstructive_tokenization domain"
            )
        tokenized = tokenization_override
    else:
        if domain_override is None:
            domain = _infer_reconstructive_domain(data)
        else:
            domain_value = domain_override.strip().lower()
            if domain_value not in {"text", "json", "logs"}:
                raise ValueError("reconstructive_domain must be one of: text, json, logs")
            domain = domain_value  # type: ignore[assignment]

        tokenized = tokenizer.tokenize(data, domain=domain)
    normalized = normalizer.normalize_tokens(tokenized.tokens)

    graph = build_deterministic_hypergraph(tokenized.tokens)
    coupling = build_coupling_matrix_summary(tokenized.tokens)
    state = fit_compact_manifold_state(normalized.points, graph)

    seed = manifold_seed_vector(state)
    # Normalize seed into the same bounded phase range used by K_M.
    seed_max = float(np.max(np.abs(seed))) if seed.size else 0.0
    if seed_max > np.pi and seed_max > 0:
        seed = seed * (np.pi / seed_max)

    fp = iterate_fixedpoint(seed)

    fidelity_bundle = _build_reconstructive_fidelity_bundle(
        state=state,
        fixedpoint=fp,
        coupling=coupling,
    )

    metadata: dict[str, str] = {
        **tokenized.metadata,
        **normalized.metadata,
        **manifold_metadata(state),
        **coupling_matrix_metadata(coupling),
        "reconstructive_token_count": str(len(tokenized.tokens)),
        "reconstructive_graph_hash": graph.graph_hash,
        "km_residual_max": f"{fp.diagnostics.residual_max:.17g}",
        "km_residual_mean": f"{fp.diagnostics.residual_mean:.17g}",
        "km_iters_mean": f"{fp.diagnostics.iterations_mean:.17g}",
        "km_valid_ratio": f"{fp.diagnostics.converged_ratio:.17g}",
        "km_seed_vector": ",".join(f"{x:.17g}" for x in seed.tolist()),
        "km_kappa": f"{KAPPA:.17g}",
        "km_eta": f"{ETA:.17g}",
        "km_tol": f"{FIXEDPOINT_TOL:.17g}",
        "km_max_iter": str(FIXEDPOINT_MAX_ITER),
        "km_threshold_residual_max": f"{fp.diagnostics.residual_max:.17g}",
        "km_threshold_valid_ratio": f"{fp.diagnostics.converged_ratio:.17g}",
        **fidelity_bundle,
    }

    full_metadata = {"preprocessing_mode": "reconstructive", **metadata}
    if domain in {"text", "json", "logs"}:
        try:
            program_source_text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                "reconstructive mode currently supports UTF-8 Text/JSON/Logs only"
            ) from exc
    else:
        program_source_text = tokenized.canonical_text

    compact_program = fit_reconstructive_program(
        program_source_text,
        domain_kind=domain,
        coupling_density=coupling.density,
        coupling_spectral_radius=coupling.spectral_radius,
        coupling_nnz=coupling.nnz,
    )
    metadata.update(compact_program)

    # Side-channel large program payload to avoid JSON-escape overhead inside
    # reconstructive_payload_v1 while keeping decode contract backward-compatible.
    program_payload = metadata.get("reconstructive_program_payload", "")
    if isinstance(program_payload, str) and program_payload:
        metadata["rpb"] = program_payload
        metadata["reconstructive_program_payload"] = "@"

    full_metadata = {"preprocessing_mode": "reconstructive", **metadata}
    validate_reconstructive_metadata(full_metadata)
    metadata[_RECONSTRUCTIVE_PAYLOAD_KEY_SHORT] = build_reconstructive_payload_metadata(
        full_metadata
    )
    # Top-level program fields are no longer needed after payload bundle construction.
    metadata.pop("reconstructive_program_type", None)
    metadata.pop("reconstructive_program_payload", None)
    return metadata


def _select_execution_mode(
    *,
    task_count: int,
    effective_workers: int,
    estimated_cost: float,
    n_strands: int,
    generators_per_block: int,
) -> str:
    """Choose serial/thread/process execution based on workload regime."""
    if task_count <= 1 or effective_workers <= 1:
        return "serial"

    complexity_signal = (n_strands * max(generators_per_block, 1)) / 16.0

    # Small or low-cost tasks are typically IPC-dominated.
    if task_count <= 8 or estimated_cost < 8.0 or complexity_signal < 2.0:
        return "serial"

    # Medium-cost tasks benefit from shared-memory threading first.
    if estimated_cost < 28.0 or task_count < (effective_workers * 2):
        return "thread"

    # High-cost tasks justify process-level parallelism.
    return "process"


def _choose_batch_size(
    *,
    task_count: int,
    effective_workers: int,
    estimated_cost: float,
    exec_mode: str,
) -> int:
    """Choose batch granularity to reduce scheduler overhead."""
    if task_count <= 1:
        return 1
    if exec_mode == "serial":
        return max(1, task_count)

    if exec_mode == "thread":
        base = 32 if estimated_cost < 16.0 else 16
    else:  # process
        base = 24 if estimated_cost < 32.0 else 12

    # Keep enough batches to distribute across workers.
    max_reasonable = max(1, task_count // max(effective_workers, 1))
    return max(1, min(base, max_reasonable))


def _build_layer_aware_batches(
    tasks_with_layer: list[tuple[int, _BlockTask]],
    *,
    batch_size: int,
) -> list[list[_BlockTask]]:
    """Build contiguous batches while respecting topology layer locality."""
    if not tasks_with_layer:
        return []

    batches: list[list[_BlockTask]] = []
    current_layer = tasks_with_layer[0][0]
    current_batch: list[_BlockTask] = []

    for layer, task in tasks_with_layer:
        if layer != current_layer or len(current_batch) >= batch_size:
            if current_batch:
                batches.append(current_batch)
            current_batch = []
            current_layer = layer
        current_batch.append(task)

    if current_batch:
        batches.append(current_batch)

    return batches
