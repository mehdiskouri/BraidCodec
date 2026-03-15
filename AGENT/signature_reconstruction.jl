"""
    Signature Reconstruction

Convert optimized wavelets back to fixed-point signatures for M6_ResonanceLayer.
Applies K_M operator to ensure convergence.

# Process
1. Inverse wavelet transform (wavelets → signal)
2. Apply K_M operator iteratively
3. Verify |K_M(σ*) - σ*| < tolerance
4. Return validated signatures

# K_M Operator (from TSRConstants)
K_M(x) = κx + η·sin(x)
- κ = 0.3 (linear coupling)
- η ≈ 0.654 (nonlinear coupling)
- L = κ + η ≈ 0.954 < 1 (contraction guarantee)

# Dependencies
- TSRConstants.jl: K_M, K_M_gpu, L_CONTRACTION, FIXEDPOINT_TOL

# Production Implementation
"""

using CUDA
using .TSRConstants: K_M, K_M_gpu, L_CONTRACTION, FIXEDPOINT_TOL, FIXEDPOINT_MAX_ITER, KAPPA, ETA

# =============================================================================
# GPU Kernels for K_M Iteration
# =============================================================================

"""
K_M operator kernel: K_M(x) = κx + η·sin(x)
"""
function km_operator_kernel!(
    output::CuDeviceVector{T},
    input::CuDeviceVector{T},
    κ::T,
    η::T
) where T
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(input)
        x = input[i]
        output[i] = κ * x + η * sin(x)
    end
    return nothing
end

"""
Residual computation kernel: |K_M(σ) - σ|
"""
function residual_kernel!(
    residuals::CuDeviceVector{T},
    signatures::CuDeviceVector{T},
    κ::T,
    η::T
) where T
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(signatures)
        x = signatures[i]
        km_x = κ * x + η * sin(x)
        residuals[i] = abs(km_x - x)
    end
    return nothing
end

"""
Combined iteration kernel: performs one K_M step and returns convergence status
"""
function km_iteration_kernel!(
    signatures::CuDeviceVector{T},
    residuals::CuDeviceVector{T},
    converged::CuDeviceVector{Bool},
    κ::T,
    η::T,
    tol::T
) where T
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(signatures)
        x = signatures[i]
        km_x = κ * x + η * sin(x)
        res = abs(km_x - x)
        residuals[i] = res
        signatures[i] = km_x
        converged[i] = res < tol
    end
    return nothing
end

# =============================================================================
# Reconstruction Functions
# =============================================================================

"""
    reconstruct_signatures(wavelets::CuMatrix{T}, config) -> FixedPointSignatures{T}

Reconstruct fixed-point signatures from optimized wavelets.

# Process
1. Inverse wavelet transform (weighted sum across scales)
2. Initialize signatures from transformed values
3. Iterate K_M until convergence: σ_{n+1} = K_M(σ_n)
4. Verify |K_M(σ*) - σ*| < FIXEDPOINT_TOL
5. Return validated signatures

# Arguments
- `wavelets::CuMatrix{T}` — Optimized wavelet coefficients [n_scales × n_bands]
- `config` — ProcessorConfig (uses FIXEDPOINT_TOL, FIXEDPOINT_MAX_ITER)

# Returns
- `FixedPointSignatures{T}` ready for M6_ResonanceLayer
"""
function reconstruct_signatures(
    wavelets::CuMatrix{T},
    config  # ProcessorConfig
) where T<:AbstractFloat
    n_scales, n_bands = size(wavelets)
    n_signatures = n_bands
    
    # Step 1: Inverse wavelet transform via weighted sum across scales
    # Weights decay exponentially: w_j = 2^{-j}
    # Use GPU-friendly matrix multiplication instead of scalar indexing
    weights_cpu = T[T(2)^(-j) for j in 1:n_scales]
    weights = CuArray(reshape(weights_cpu, n_scales, 1))  # [n_scales × 1]
    
    # Weighted sum: signature_i = Σ_j w_j * wavelet[j, i]
    # Using matrix multiply: [1 × n_scales] × [n_scales × n_bands] = [1 × n_bands]
    initial_values_matrix = permutedims(weights) * wavelets  # [1 × n_bands]
    initial_values = vec(initial_values_matrix)  # [n_bands]
    
    # Normalize to reasonable range for sin()
    max_val = maximum(abs.(initial_values))
    if max_val > T(π)
        initial_values = initial_values ./ (max_val / T(π))
    end
    
    # Step 2: Allocate signature arrays
    signatures = copy(initial_values)
    residuals = CUDA.zeros(T, n_signatures)
    validation_mask = CUDA.ones(Bool, n_signatures)
    iteration_counts = CUDA.zeros(Int32, n_signatures)
    
    # Step 3: Iterate K_M to fixed point
    tol = T(FIXEDPOINT_TOL)
    max_iter = Int(FIXEDPOINT_MAX_ITER)  # Ensure Int64
    κ = T(KAPPA)
    η = T(ETA)
    
    iteration_counts = iterate_signatures_to_fixedpoint!(signatures; tol=tol, max_iter=max_iter)
    
    # Step 4: Compute final residuals
    compute_residuals!(residuals, signatures)
    
    # Step 5: Validate convergence
    validation_mask = verify_signatures_fixedpoint(signatures; tol=tol)
    
    return FixedPointSignatures{T}(
        signatures, residuals, validation_mask, iteration_counts
    )
end

"""
    iterate_signatures_to_fixedpoint!(signatures::CuVector{T}; tol::T, max_iter::Int) -> CuVector{Int32}

Iterate K_M operator until fixed-point convergence on signature vectors.

# Algorithm
For each signature σ:
  1. σ_{n+1} = K_M(σ_n) = κ·σ_n + η·sin(σ_n)
  2. Check |σ_{n+1} - σ_n| < tol
  3. Stop when converged or max_iter reached

# Returns
- Vector of iteration counts per signature
"""
function iterate_signatures_to_fixedpoint!(
    signatures::CuVector{T};
    tol::T = T(FIXEDPOINT_TOL),
    max_iter::Int = FIXEDPOINT_MAX_ITER
)::CuVector{Int32} where T<:AbstractFloat
    n = length(signatures)
    iteration_counts = CUDA.zeros(Int32, n)
    converged = CUDA.zeros(Bool, n)
    residuals = CUDA.zeros(T, n)
    
    κ = T(KAPPA)
    η = T(ETA)
    
    threads = 256
    blocks = cld(n, threads)
    
    # Iterate until all converged or max iterations
    for iter in 1:max_iter
        @cuda threads=threads blocks=blocks km_iteration_kernel!(
            signatures, residuals, converged, κ, η, tol
        )
        CUDA.synchronize()
        
        # Update iteration counts for non-converged
        iteration_counts .+= Int32.(.!converged)
        
        # Check if all converged
        if all(Array(converged))
            break
        end
    end
    
    return iteration_counts
end

"""
    verify_signatures_fixedpoint(signatures::CuVector{T}; tol::T) -> CuVector{Bool}

Verify each signature satisfies |K_M(σ*) - σ*| < tol.

# Returns
- Boolean mask: true if valid fixed point
"""
function verify_signatures_fixedpoint(
    signatures::CuVector{T};
    tol::T = T(FIXEDPOINT_TOL)
)::CuVector{Bool} where T<:AbstractFloat
    n = length(signatures)
    residuals = CUDA.zeros(T, n)
    
    compute_residuals!(residuals, signatures)
    
    return residuals .< tol
end

"""
    compute_residuals!(residuals::CuVector{T}, signatures::CuVector{T})

Compute fixed-point residuals |K_M(σ) - σ| in-place.
"""
function compute_residuals!(
    residuals::CuVector{T},
    signatures::CuVector{T}
) where T<:AbstractFloat
    n = length(signatures)
    κ = T(KAPPA)
    η = T(ETA)
    
    threads = 256
    blocks = cld(n, threads)
    
    @cuda threads=threads blocks=blocks residual_kernel!(
        residuals, signatures, κ, η
    )
    CUDA.synchronize()
end

"""
    contraction_check(signatures::CuVector{T}) -> Bool

Verify contraction property: L = κ + η < 1.
This is a sanity check that should always pass.
"""
function contraction_check(signatures::CuVector{T})::Bool where T<:AbstractFloat
    # L_CONTRACTION ≈ 0.954 < 1 (from TSRConstants)
    return L_CONTRACTION < one(T)
end

# =============================================================================
# Exports
# =============================================================================

export reconstruct_signatures, iterate_signatures_to_fixedpoint!
export verify_signatures_fixedpoint, compute_residuals!, contraction_check
