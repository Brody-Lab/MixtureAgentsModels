###############
### MoA-HMM ###
###############
"""
    ModelHMM{T1<:AbstractArray, T2<:AbstractVector, T3<:AbstractArray} <: MixtureAgentsModel. 

Model parameters for an MoA-HMM

Fields:
- `β`: (nagents x nstates) matrix of agent weights
- `π`: (nstates x 1) vector of initial state probabilities
- `A`: (nstates x nstates) matrix of state transition probabilities
"""
@with_kw struct ModelHMM{T1,T2,T3} <: MixtureAgentsModel where {T1 <: AbstractArray, T2 <: AbstractVector, T3 <: AbstractArray}
    β::T1
    π::T2
    A::T3
end

"""
    ModelOptionsHMM <: ModelOptions

Parameters for fitting an MoA-HMM

Fields:
- `nstates`: number of latent states to fit
- `β0`: (1 x nstates) or (nagents x nstates) matrix of initial agent weights
- `βprior`: boolean indicating whether to use a prior distribution for β. Prior distribution used specified by agent. Defaults to true
- `A0`: (nstates x nstates) matrix of initial state transition probabilities
- `α_A`: (nstates x nstates) matrix of shaping parameters for Dirichlet prior on state transition prior distributions. Defaults to ones, which is equivalent to no prior.
- `π0`: (nstates x 1) vector of initial state probabilities
- `α_π`: (nstates x 1) vector of shaping parameters for Dirichlet prior on initial state probabilities. Defaults to ones, which is equivalent to no prior.
- `nstarts`: number of initializations for gradient descent
- `maxiter`: maximum number of EM iterations
- `tol`: the difference in likelihood and/or difference in parameter values required for EM termination
"""
@with_kw struct ModelOptionsHMM{T<:AbstractArray,P1<:AbstractVector,P2<:AbstractVector,A1<:AbstractArray,A2<:AbstractArray,I<:Int,F<:Float64,B<:Bool} <: ModelOptions 
    nstates::I = 2
    β0::T = zeros(SizedMatrix{1,nstates})
    βprior::B = true
    π0::P1 = zeros(SizedVector{nstates})
    α_π::P2 = ones(SizedVector{nstates})
    A0::A1 = zeros(SizedMatrix{nstates,nstates})
    α_A::A2 = ones(SizedMatrix{nstates,nstates})
    nstarts::I = 1
    maxiter::I = 300
    tol::F = 1E-5
end

"""
    ModelOptions(model::ModelHMM;kwargs...)

Make ModelOptions with initial values from input model.
kwargs specifies additional keyword arguments for ModelOptions
"""
function ModelOptions(model::ModelHMM;kwargs...)
    β0 = model.β
    A0 = model.A
    π0 = model.π
    nstates = size(β0,2)
    return ModelOptionsHMM(β0=β0,A0=A0,π0=π0,nstates=nstates;kwargs...)
end

"""
    ModelOptions(model::ModelHMM,options::ModelOptionsHMM)

Make ModelOptions with initial values from input model and other settings from existing ModelOptions
"""
function ModelOptions(model::ModelHMM,options::ModelOptionsHMM)
    @unpack nstarts,maxiter,tol,βprior,α_π,α_A = options
    β0 = model.β
    A0 = model.A
    π0 = model.π

    nstates = size(β0,1)
    return ModelOptionsHMM(β0=β0,A0=A0,π0=π0,nstates=nstates,nstarts=nstarts,maxiter=maxiter,tol=tol, βprior=βprior, α_π=α_π, α_A=α_A)
end

"""
    ModelOptions(β0::Array{Float64},options::ModelOptionsHMM)

Make ModelOptions using `β0`, `A0`, and `π0` and other settings from existing ModelOptions
"""
function ModelOptions(β0::AbstractArray,π0::AbstractVector,A0::AbstractArray,options::ModelOptionsHMM)
    @unpack nstates,nstarts,maxiter,tol,βprior,α_π,α_A = options
    return ModelOptionsHMM(β0=β0,A0=A0,π0=π0,βprior=βprior,α_π=α_π,α_A=α_A,nstates=nstates,nstarts=nstarts,maxiter=maxiter,tol=tol)
end

"""
    ModelOptions(β0::Array{Float64},options::ModelOptionsHMM)

Make ModelOptions using `β0`, `A0`, and `π0` and other settings from existing ModelOptions
"""
function ModelOptions(β0::AbstractArray,π0::AbstractVector,A0::AbstractArray,α_π::AbstractVector,α_A::AbstractArray, options::ModelOptionsHMM)
    @unpack nstates,nstarts,maxiter,tol,βprior = options
    return ModelOptionsHMM(β0=β0,π0=π0,A0=A0,α_π=α_π,α_A=α_A,nstates=nstates,nstarts=nstarts,maxiter=maxiter,tol=tol,βprior=βprior)
end

"""
    modeltype(options::ModelOptionsHMM)

Return model type corresponding to `options`
"""
function modeltype(options::ModelOptionsHMM)
    return ModelHMM
end

#################
### MoA-Drift ###
#################
"""
    MixtureAgentsModelβ{T1<:Real, T2<:Real} 

Model parameters for an MoA-β(drift) model
    
Fields:
- `β`: (nagents x ntrials) matrix of agent weights
- `σ`: (nagents x 1) vector of standard deviations for each agents' noisy drift on β
- `σInit`: (nagents x 1) vector of standard deviations for each agents' initial drift on β
- `σSess`: (nagents x 1) vector of standard deviations for each agents' session drift on β
"""
@with_kw struct ModelDrift{T1,T2,T3,T4} <: MixtureAgentsModel where {T1 <: AbstractArray, T2 <: AbstractVector, T3 <: AbstractVector, T4 <: AbstractVector}
    β::T1
    σ::T2
    σInit::T3
    σSess::T4
end

"""
    ModelOptionsDrift <: ModelOptions

Parameters for fitting an MoA-drift model

Fields:
- `σ0`: (nagents x 1) vector of initial noise standard deviations
- `σInit0`: (nagents x 1) vector of initial noise standard deviations for initial drift
- `σSess0`: (nagents x 1) vector of initial noise standard deviations for session drift
- `nstarts`: number of initializations for gradient descent
- `maxiter`: maximum number of EM iterations
- `tol`: the difference in likelihood required for completion
"""
@with_kw struct ModelOptionsDrift{S<:AbstractVector,SI<:AbstractVector,SS<:AbstractVector,I<:Int,F<:Float64} <: ModelOptions 
    σ0::S = [0.]
    σInit0::SI = [0.]
    σSess0::SS = [0.]
    nstarts::I = 1
    maxiter::I = 1000
    tol::F = 1E-5
end

"""
    ModelOptions(model::ModelDrift;kwargs...)

Make ModelOptions with initial values from input model.
kwargs specify additional keyword arguments for ModelOptions
"""
function ModelOptions(model::ModelDrift;kwargs...)
    σ0 = model.σ
    σInit0 = model.σInit
    σSess0 = model.σSess
    return ModelOptionsDrift(σ0=σ0,σInit0=σInit0,σSess0=σSess0;kwargs...)
end

"""
    ModelOptions(model::ModelDrift,options::ModelOptionsDrift)

Make ModelOptions with initial values from input model and other settings from existing ModelOptions
"""
function ModelOptions(model::ModelDrift,options::ModelOptionsDrift)
    @unpack nstarts,maxiter,tol = options
    σ0 = model.σ
    σInit0 = model.σInit
    σSess0 = model.σSess
    return ModelOptionsDrift(σ0=σ0,σInit0=σInit0,σSess0=σSess0,nstarts=nstarts,maxiter=maxiter,tol=tol)
end

"""
    ModelOptions(σ0,σInit0,σSess0,options::ModelOptionsDrift)

Make ModelOptions with `σ0`, `σInit0`, and `σSess0` and other settings from existing ModelOptions
"""
function ModelOptions(σ0,σInit0,σSess0,options::ModelOptionsDrift)
    @unpack nstarts,maxiter,tol = options
    return ModelOptionsDrift(σ0=σ0,σInit0=σInit0,σSess0=σSess0,nstarts=nstarts,maxiter=maxiter,tol=tol)
end


