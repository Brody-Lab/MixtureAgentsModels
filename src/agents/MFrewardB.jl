"""
    θMFrewardB{T} <: Agent

Parameters for an agent computing values as a model-free reward kernel from an Ito & Doya model-based reinforcement learning framework. Unlike MFreward, the learning rate `α` on reward is absorbed into the agent weight `β` to reduce interactions between the two parameters.

Fields:
- `Qinit`: initial Q values
- `α`: learning rate. randomly initialized by default from a Beta distribution
- `θL2`: standard deviation of normal distribution to initialize GLM weights
- `color`: color of agent for plotting
- `color_lite`: lighter color of agent for plotting
"""
@with_kw struct MFrewardB{T<:Real,Q<:AbstractVector,I<:Int,B<:AbstractVector,A<:Distribution,C<:Symbol} <: Agent 
    Q0::Q = @SVector [0.5,0.5,0,0]
    α::T = rand(Beta(5,5))
    αscale::I = 1
    βprior::B = [Normal(0,10)]
    αprior::A = Beta(1,1)
    color::C = :gold3
    color_lite::C = :mediumaquamarine
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
function next_Q!(Q::AbstractArray,θ::MFrewardB,data::D,t::Int) where D <: RatData
    @unpack α = θ
    @unpack choices,rewards = data
    Q .*= (1 - α)
    Q[choices[t]] += rewards[t]
end

"""
    αtitle(θ)

Helper function for plotting
"""
function αtitle(θ::MFrewardB)
    return "α(MFr)"
end

"""
    βtitle(θ)

Helper function for plotting
"""
function βtitle(θ::MFrewardB)
    return "β(MFr)"
end

function atick(θ::MFrewardB)
    return "MFr"
end


"""
    agent2string(θ)

Gets string corresponding to agent
"""
function agent2string(θ::MFrewardB)
    return "MFrewardB"
end

