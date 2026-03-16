"""Encoder — full pipeline from bytes to ``EncodedStream``.

Maps raw data through chunking → braid construction → invariant computation
→ optional simplification → ``EncodedStream`` assembly.  Parallel block
encoding via ``ProcessPoolExecutor`` for multi-block payloads.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import time
import zlib
from base64 import b85encode
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import TYPE_CHECKING, cast

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
from braidcodec.codec.equation_discovery import (
    compile_discovered_braid_equation_program,
    compile_discovered_equation_program,
    discover_equations,
)
from braidcodec.codec.equation_library import (
    CandidateLibraryEntry,
    CandidateLibraryStore,
    compute_semantic_signature,
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
    compute_bfps,
    mean_estimated_cost,
    morton_key_1d,
    sort_chunk_indices_for_topology,
    synthesize_topology_generators_v2,
    topology_commitment_v2,
)
from braidcodec.codec.reconstructive_compact import (
    fit_reconstructive_program,
    synthesize_reconstructive_bytes,
)
from braidcodec.codec.reconstructive_transform import reconstructive_forward_generators
from braidcodec.codec.schema import (
    EncodedBlock,
    EncodedStream,
    build_reconstructive_payload_metadata,
    compute_reconstructive_commitment_v3,
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
_VALID_RECONSTRUCTIVE_COMPACT_TRANSPORT: frozenset[str] = frozenset(
    {"enabled", "lean"}
)
_PROGRAM_TYPE_TO_TRANSPORT_CODE: dict[str, str] = {
    "discovered-braid-equation-v1": "dbe1",
    "discovered-equation-v1": "de1",
    "repeat-text-v1": "rt1",
    "json-linear-items-v1": "jli1",
    "json-literal-v1": "jl1",
    "logs-seq-v1": "ls1",
    "latent-residual-v2": "lr2",
    "latent-residual-v3": "lr3",
}
_COMPACT_SIDECHANNEL_PACK_PREFIX = "~sp85:"
_COMPACT_RECON_HEADER_PREFIX = "~rh85:"
_COMPACT_AUDIT_BUNDLE_PREFIX = "~ra85:"
_RECONSTRUCTIVE_AUDIT_KEY_SHORT = "ra1"
_RECONSTRUCTIVE_AUDIT_KEYS: tuple[str, ...] = (
    "batch_count",
    "batch_size",
    "coupling_matrix_density",
    "coupling_matrix_hash",
    "coupling_matrix_nnz",
    "coupling_matrix_spectral_radius",
    "discovery_profile_id",
    "equation_library_hit",
    "equation_library_hit_mode",
    "equation_library_signature",
    "equation_symbolic_hash",
    "estimated_cost_mean",
    "execution_mode",
    "fidelity_bundle_v1",
    "fidelity_coherence",
    "fidelity_energy",
    "fidelity_topology",
    "km_iters_mean",
    "km_residual_max",
    "km_residual_mean",
    "km_valid_ratio",
    "task_count",
    "timing_encode_core_s",
    "timing_layer_order_s",
    "timing_preprocess_s",
    "timing_total_s",
)


def _transport_family(code: str) -> str:
    """Return transport family from compact code (e.g. ps1 from ps1.dbe1)."""
    if "." in code:
        return code.split(".", 1)[0]
    return code


def _pack_compact_sidechannel_blob(blob: str) -> str:
    """Pack compact sidechannel blob when shorter than raw payload."""
    if not blob:
        return blob
    packed = _COMPACT_SIDECHANNEL_PACK_PREFIX + b85encode(zlib.compress(blob.encode("utf-8"), level=9)).decode(
        "ascii"
    )
    return packed if len(packed) < len(blob) else blob


def _pack_compact_reconstructive_header(*, transport_code: str, sidechannel_blob: str) -> str:
    """Pack compact reconstructive transport header into one metadata field."""
    msgpack_mod = importlib.import_module("msgpack")
    packed_obj = msgpack_mod.packb(
        {
            "t": transport_code,
            "p": sidechannel_blob,
        },
        use_bin_type=True,
    )
    packed = _COMPACT_RECON_HEADER_PREFIX + b85encode(zlib.compress(bytes(packed_obj), level=9)).decode("ascii")

    legacy_json = json.dumps(
        {
            "t": transport_code,
            "p": sidechannel_blob,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return packed if len(packed) < len(legacy_json) else legacy_json


def _pack_compact_reconstructive_audit_bundle(audit_bundle: dict[str, str]) -> str:
    """Pack optional compact reconstructive audit bundle into one metadata field."""
    payload = json.dumps(audit_bundle, sort_keys=True, separators=(",", ":"))
    packed = _COMPACT_AUDIT_BUNDLE_PREFIX + b85encode(zlib.compress(payload.encode("utf-8"), level=9)).decode(
        "ascii"
    )
    return packed if len(packed) < len(payload) else payload


def _build_compact_reconstructive_audit_bundle(metadata: dict[str, str]) -> dict[str, str]:
    """Build optional audit bundle for compact reconstructive streams."""
    audit: dict[str, str] = {}
    for key in _RECONSTRUCTIVE_AUDIT_KEYS:
        value = metadata.get(key, "")
        if value:
            audit[key] = value
    return audit


def _select_compact_transport_metadata(*, transport_code: str, sidechannel_blob: str) -> dict[str, str]:
    """Choose the smallest metadata representation for compact transport."""
    header = _pack_compact_reconstructive_header(
        transport_code=transport_code,
        sidechannel_blob=sidechannel_blob,
    )
    # Simple envelope estimate: key+value bytes. Wire framing overhead is similar
    # enough for these tiny maps that this reliably picks the smaller form.
    header_est = len("rh") + len(header)
    explicit_est = len("rt") + len(transport_code) + len("rpb") + len(sidechannel_blob)
    if header_est < explicit_est:
        return {"rh": header}
    return {
        "rt": transport_code,
        "rpb": sidechannel_blob,
    }


def _resolve_km_thresholds(
    *,
    profile_id: str,
    residual_max_observed: float,
    valid_ratio_observed: float,
) -> tuple[float, float, bool]:
    """Resolve K_M gate thresholds for a given profile.

    Returns `(residual_max_threshold, valid_ratio_threshold, enforce_profile)`.
    """
    normalized = profile_id.strip().lower()
    if normalized in {"", "default", "default-v1"}:
        return residual_max_observed, valid_ratio_observed, False
    if normalized == "strict-v1":
        return 1e-3, 0.9, True
    raise ValueError("reconstructive_strict_gate_profile must be one of: default-v1, strict-v1")

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
    """Return reconstructive payload blob from current compact key."""
    return metadata.get(_RECONSTRUCTIVE_PAYLOAD_KEY_SHORT, "")


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
    reconstructive_discovery: str = "enabled",
    reconstructive_library_path: str | None = None,
    reconstructive_strict_gate_profile: str = "default-v1",
    reconstructive_compact_transport: str = "enabled",
    reconstructive_compact_audit_bundle: bool = False,
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
    reconstructive_compact_transport:
        Compact transport policy for reconstructive mode. Supported values are
        ``"enabled"`` (commitment-validated compact transport) and ``"lean"``
        (checksum-authoritative compact transport).
    reconstructive_compact_audit_bundle:
        Include optional compact audit sidecar metadata (``ra1``) for
        reconstructive compact transport modes.

    Returns
    -------
    EncodedStream
        Complete encoded payload ready for wire serialization.
    """
    if preprocessing_mode not in _VALID_PREPROCESSING_MODES:
        valid = ", ".join(sorted(_VALID_PREPROCESSING_MODES))
        raise ValueError(f"preprocessing_mode must be one of: {valid}")

    compact_mode = reconstructive_compact_transport.strip().lower()
    if compact_mode == "julia":
        compact_mode = "lean"
    if compact_mode not in _VALID_RECONSTRUCTIVE_COMPACT_TRANSPORT:
        valid = ", ".join(sorted(_VALID_RECONSTRUCTIVE_COMPACT_TRANSPORT))
        raise ValueError(f"reconstructive_compact_transport must be one of: {valid}")

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
            discovery_mode=reconstructive_discovery,
            discovery_profile_id=reconstructive_strict_gate_profile,
            library_path=reconstructive_library_path,
        )
        (
            reconstructive_meta,
            compact_transport_code,
            compact_transport_sidechannel,
        ) = _compact_reconstructive_metadata(
            reconstructive_meta,
            compact_mode=compact_mode,
            include_audit_bundle=reconstructive_compact_audit_bundle,
        )
        reconstructive_seed_vector = reconstructive_meta.get("km_seed_vector")

        if not compact_transport_code:
            raise ValueError("reconstructive compact transport payload unavailable")

        compact_blocks: tuple[EncodedBlock, ...] = tuple()
        metadata = {
            **reconstructive_meta,
        }
        transport_code = compact_transport_code
        transport_family = _transport_family(transport_code)
        commitment_sidechannel = compact_transport_sidechannel or str(metadata.get("rpb", ""))
        if transport_family != "ps2":
            metadata["rc3"] = compute_reconstructive_commitment_v3(
                transport_code=transport_code,
                blocks=compact_blocks,
                program_sidechannel_blob=commitment_sidechannel,
            )

        return EncodedStream(
            blocks=compact_blocks,
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
        # Non-topology modes do not emit BFPS profiles, but still need a
        # realistic cost signal so large payloads can parallelize.
        estimated_cost = (key.n_strands * max(generators_per_block, 1)) / 16.0
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


def _compact_reconstructive_metadata(
    metadata: dict[str, str],
    *,
    compact_mode: str,
    include_audit_bundle: bool,
) -> tuple[dict[str, str], str, str]:
    """Collapse reconstructive metadata to enabled/lean compact transport form."""

    payload_json = _get_reconstructive_payload_blob(metadata)
    if not payload_json:
        raise ValueError("reconstructive compact transport payload missing")

    try:
        payload_obj = json.loads(payload_json)
    except Exception:
        raise ValueError("reconstructive compact transport payload invalid")

    if not isinstance(payload_obj, dict):
        raise ValueError("reconstructive compact transport payload shape invalid")

    payload_obj_typed = cast("dict[str, object]", payload_obj)
    program_type = str(payload_obj_typed.get("reconstructive_program_type", ""))
    sidechannel = str(metadata.get("rpb", ""))
    if not program_type or not sidechannel:
        raise ValueError("reconstructive compact transport payload unavailable")

    type_code = _PROGRAM_TYPE_TO_TRANSPORT_CODE.get(program_type, "")
    if not type_code:
        raise ValueError("reconstructive compact transport program type unsupported")

    packed_sidechannel = _pack_compact_sidechannel_blob(sidechannel)
    audit_bundle = _build_compact_reconstructive_audit_bundle(metadata) if include_audit_bundle else {}

    if compact_mode == "lean":
        transport_code = f"ps2.{type_code}"
        compact_meta = _select_compact_transport_metadata(
            transport_code=transport_code,
            sidechannel_blob=packed_sidechannel,
        )
        if audit_bundle:
            compact_meta[_RECONSTRUCTIVE_AUDIT_KEY_SHORT] = _pack_compact_reconstructive_audit_bundle(audit_bundle)
        return (
            compact_meta,
            transport_code,
            packed_sidechannel,
        )

    transport_code = f"ps1.{type_code}"
    compact_meta = _select_compact_transport_metadata(
        transport_code=transport_code,
        sidechannel_blob=packed_sidechannel,
    )
    if audit_bundle:
        compact_meta[_RECONSTRUCTIVE_AUDIT_KEY_SHORT] = _pack_compact_reconstructive_audit_bundle(audit_bundle)
    return (
        compact_meta,
        transport_code,
        packed_sidechannel,
    )


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
    discovery_mode: str = "enabled",
    discovery_profile_id: str = "default-v1",
    library_path: str | None = None,
) -> dict[str, str]:
    """Compute deterministic reconstructive metadata contract payload."""
    tokenizer = FrequencyTokenizerV1()
    normalizer = OscillatorNormalizationV1()

    if tokenization_override is not None:
        domain: DomainKind = tokenization_override.domain
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
            domain = cast("DomainKind", domain_value)

        tokenized = tokenizer.tokenize(data, domain=domain)
    normalized = normalizer.normalize_tokens(tokenized.tokens)

    graph = build_deterministic_hypergraph(tokenized.tokens)
    coupling = build_coupling_matrix_summary(tokenized.tokens)
    state = fit_compact_manifold_state(normalized.points, graph)

    # Keep reconstructive path topology-coupled with a lightweight binary
    # fixed-point signature over source bytes.
    topo_sig = compute_bfps(data)
    topo_morton = morton_key_1d(len(tokenized.tokens), lane=0)
    topo_commitment = topology_commitment_v2(
        morton_key=topo_morton,
        layer_index=0,
        hash32=topo_sig.hash32,
    )

    seed = manifold_seed_vector(state)
    # Normalize seed into the same bounded phase range used by K_M.
    seed_max = float(np.max(np.abs(seed))) if seed.size else 0.0
    if seed_max > np.pi and seed_max > 0:
        seed = seed * (np.pi / seed_max)

    fp = iterate_fixedpoint(seed)

    km_threshold_residual_max, km_threshold_valid_ratio, enforce_profile = _resolve_km_thresholds(
        profile_id=discovery_profile_id,
        residual_max_observed=float(fp.diagnostics.residual_max),
        valid_ratio_observed=float(fp.diagnostics.converged_ratio),
    )

    if enforce_profile:
        if fp.diagnostics.residual_max > km_threshold_residual_max:
            raise ValueError(
                "reconstructive strict gate failed: residual exceeds threshold "
                f"({fp.diagnostics.residual_max:.6g} > {km_threshold_residual_max:.6g})"
            )
        if fp.diagnostics.converged_ratio < km_threshold_valid_ratio:
            raise ValueError(
                "reconstructive strict gate failed: valid ratio below threshold "
                f"({fp.diagnostics.converged_ratio:.6g} < {km_threshold_valid_ratio:.6g})"
            )

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
        "km_threshold_residual_max": f"{km_threshold_residual_max:.17g}",
        "km_threshold_valid_ratio": f"{km_threshold_valid_ratio:.17g}",
        "discovery_profile_id": discovery_profile_id,
        "topology_hash32": str(topo_sig.hash32),
        "topology_density_fp": str(topo_sig.density_fp),
        "topology_centroid_fp": str(topo_sig.centroid_fp),
        "topology_variance_fp": str(topo_sig.variance_fp),
        "topology_morton_key": str(topo_morton),
        "topology_commitment": str(topo_commitment),
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

    mode = discovery_mode.strip().lower()
    if mode not in {"enabled", "disabled", "required"}:
        raise ValueError("reconstructive_discovery must be one of: enabled, disabled, required")

    compact_program: dict[str, str] | None = None
    discovery_error_reason = ""

    signature = compute_semantic_signature(
        {
            "domain_kind": domain,
            "tokenizer_id": metadata["tokenizer_id"],
            "tokenizer_version": metadata["tokenizer_version"],
            "bin_table_hash": metadata["bin_table_hash"],
            "normalization_profile_hash": metadata["normalization_profile_hash"],
            "manifold_state_hash": metadata["manifold_state_hash"],
            "reconstructive_graph_hash": metadata["reconstructive_graph_hash"],
            "topology_hash32": metadata["topology_hash32"],
            "topology_density_fp": metadata["topology_density_fp"],
            "topology_centroid_fp": metadata["topology_centroid_fp"],
            "topology_variance_fp": metadata["topology_variance_fp"],
            "topology_commitment": metadata["topology_commitment"],
        }
    )
    metadata["equation_library_signature"] = signature
    metadata["equation_library_hit"] = "0"
    metadata["equation_library_hit_mode"] = "none"

    lib = CandidateLibraryStore.from_optional_path(library_path)
    if mode != "disabled":
        lib_entry = lib.retrieve(signature) if lib is not None else None
        if lib_entry is not None and lib_entry.domain_kind == domain:
            candidate_program = {
                "reconstructive_program_type": lib_entry.reconstructive_program_type,
                "reconstructive_program_payload": lib_entry.reconstructive_program_payload,
            }
            try:
                if synthesize_reconstructive_bytes(candidate_program) == data:
                    compact_program = candidate_program
                    metadata["equation_library_hit"] = "1"
                    metadata["equation_library_hit_mode"] = "exact"
                    metadata["equation_symbolic_hash"] = lib_entry.symbolic_hash
            except Exception:
                # Ignore stale cache entries and continue with live discovery.
                compact_program = None

        if compact_program is None and lib is not None:
            for compat_entry in lib.retrieve_compatible(
                domain_kind=domain,
                exclude_signature=signature,
                limit=32,
            ):
                candidate_program = {
                    "reconstructive_program_type": compat_entry.reconstructive_program_type,
                    "reconstructive_program_payload": compat_entry.reconstructive_program_payload,
                }
                try:
                    if synthesize_reconstructive_bytes(candidate_program) == data:
                        compact_program = candidate_program
                        metadata["equation_library_hit"] = "1"
                        metadata["equation_library_hit_mode"] = "compatible"
                        metadata["equation_symbolic_hash"] = compat_entry.symbolic_hash
                        break
                except Exception:
                    # Ignore stale/invalid compatible entries and continue.
                    continue

        if compact_program is None:
            discovery = discover_equations(
                data,
                domain_kind=domain,
                profile_id=discovery_profile_id,
            )
            if discovery.equation is not None:
                try:
                    candidate_program = compile_discovered_braid_equation_program(discovery.equation)
                except Exception:
                    # Keep backward-compatible path if braid lift fails unexpectedly.
                    candidate_program = compile_discovered_equation_program(discovery.equation)
                try:
                    if synthesize_reconstructive_bytes(candidate_program) == data:
                        compact_program = candidate_program
                        metadata["equation_library_hit_mode"] = "discovery"
                        metadata["equation_symbolic_hash"] = discovery.equation.symbolic_hash
                        if lib is not None:
                            lib.promote(
                                CandidateLibraryEntry(
                                    signature=signature,
                                    domain_kind=domain,
                                    reconstructive_program_type=candidate_program[
                                        "reconstructive_program_type"
                                    ],
                                    reconstructive_program_payload=candidate_program[
                                        "reconstructive_program_payload"
                                    ],
                                    symbolic_hash=discovery.equation.symbolic_hash,
                                )
                            )
                except Exception:
                    discovery_error_reason = "discovery-non-exact"
            else:
                discovery_error_reason = "discovery-no-equation"

    if compact_program is None:
        if mode == "required":
            reason = discovery_error_reason or "discovery-required-no-exact-program"
            raise ValueError(f"reconstructive discovery required but unavailable: {reason}")

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
    if program_payload:
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

    # Small batches are scheduler/IPC-dominated.
    if task_count <= 8:
        return "serial"

    # Cheap tasks stay serial unless there are enough blocks to amortize
    # thread/process overhead.
    if estimated_cost < 8.0 and task_count < (effective_workers * 4):
        return "serial"
    if complexity_signal < 2.0 and task_count < (effective_workers * 6):
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
