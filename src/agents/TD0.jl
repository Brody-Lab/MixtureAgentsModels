"""
    TD0{T} <: Agent

Parameters for an agent computing values as a model-free (TD0) agent

Fields:
- `Qinit`: initial Q values
- `α`: learning rate. randomly initialized by default from a Beta distribution
- `γ`: discount factor. randomly initialized by default from a Beta distribution
- `θL2`: standard deviation of normal distribution to initialize GLM weights
- `color`: color of agent for plotting
- `color_lite`: lighter color of agent for plotting
"""
@with_kw struct TD0{T1<:Real,T2<:Real,Q<:AbstractVector,I<:Int,A1<:Distribution,A2<:Distribution,B<:AbstractVector,C<:Symbol} <: Agent 
    Q0::Q = @SVector [0.5,0.5,0.5,0.5]
    α::T1 = rand(Beta(5,5))
    γ::T2 = rand(Beta(5,5))
    αprior::A1 = Beta(2,2)
    αscale::I = 1
    γprior::A2 = Beta(1,1)
    γscale::I = 1
    βprior::B = [Normal(0,10)]
    color::C = :lightseagreen
    color_lite::C = :aquamarine
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
function next_Q!(Q::AbstractArray,θ::TD0,data::TwoStepData,t::Int) 
    @unpack α,γ = θ
    @unpack choices,outcomes,rewards = data
    Q[1:2] *= (1 - α)
    Q[choices[t]] += α*γ*Q[2+outcomes[t]]
    Q[3:4] *= (1 - α)
    Q[2+outcomes[t]] += α*rewards[t]
end

"""
    αtitle(θ)

Helper function for plotting
"""
function αtitle(θ::TD0)
    return "α(TD0)"
end

"""
    βtitle(θ)

Helper function for plotting
"""
function βtitle(θ::TD0)
    return "β(TD0)"
end

"""
    agent2string(θ)

Gets string corresponding to agent
"""
function agent2string(θ::TD0)
    return "TD0"
end