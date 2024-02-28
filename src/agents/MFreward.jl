"""
    MFreward{T} <: Agent

Parameters for an agent computing values as a model-free reward kernel from an Ito & Doya model-based reinforcement learning framework

Fields:
- `Qinit`: initial Q values
- `α`: learning rate. randomly initialized by default from a Beta distribution
- `θL2`: standard deviation of normal distribution to initialize GLM weights
- `color`: color of agent for plotting
- `color_lite`: lighter color of agent for plotting
"""

@with_kw struct MFreward{T<:Real,Q<:AbstractVector,I<:Int,A<:Distribution,B<:AbstractVector,C<:Symbol} <: Agent
    Q0::Q = @SVector [0.5,0.5,0.5,0.5]
    α::T = rand(Beta(5,5))
    αprior::A = Beta(1,1)
    βprior::B = [Normal(0,10)]
    αscale::I = 1
    color::C = :seagreen
    color_lite::C = :palegreen
end

"""
    next_Q(Q, θ, data, t)

Updates the Q value given its previous value, agent parameters, and trial data.

Arguments:
- `Q`: previous Q value
- `θ`: agent parameters
- `data`: behavioral data
- `t`: trial number to pull from data
"""
function next_Q!(Q::AbstractArray,θ::MFreward,data::D,t::Int) where D <: RatData
    @unpack α = θ
    @unpack choices,rewards = data
    Q .*= (1 - α)
    Q[choices[t]] += α*rewards[t]
end

"""
    αtitle(θ)

Helper function for plotting
"""
function αtitle(θ::MFreward)
    return "α(MFreward)"
end

"""
    βtitle(θ)

Helper function for plotting
"""
function βtitle(θ::MFreward)
    return "β(MFreward)"
end

"""
    agent2string(θ)

Gets string corresponding to agent
"""
function agent2string(θ::MFreward)
    return "MFreward"
end

function atick(θ::MFreward)
    return "MFr"
end
