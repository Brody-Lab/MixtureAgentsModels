"""
    MBreward{T} <: Agent

Parameters for an agent computing values as a model-based reward kernel from an Ito & Doya model-based reinforcement learning framework

Fields:
- `Qinit`: initial Q values
- `α`: learning rate. randomly initialized by default from a Beta distribution
- `θL2`: standard deviation of normal distribution to initialize GLM weights
- `color`: color of agent for plotting
- `color_lite`: lighter color of agent for plotting
"""
@with_kw struct MBreward{T<:Real,Q<:AbstractVector,I<:Int,A<:Distribution,B<:AbstractArray,C<:Symbol} <: Agent 
    Q0::Q = @SVector [0.5,0.5,0,0]
    α::T = rand(Beta(5,5))
    αprior::A = Beta(1,1)
    βprior::B = [Normal(0,10)]
    αscale::I = 1
    color::C = :purple
    color_lite::C = :plum
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
function next_Q!(Q::AbstractArray,θ::MBreward,data::TwoStepData,t::Int) 
    @unpack α = θ
    @unpack choices,nonchoices,rewards,trans_commons = data
    Q .*= (1 - α)
    if trans_commons[t]
        Q[choices[t]] += α*rewards[t]
    else
        Q[nonchoices[t]] += α*rewards[t]
    end
end

"""
    αtitle(θ)

Helper function for plotting
"""
function αtitle(θ::MBreward)
    return "α(MBreward)"
end

"""
    βtitle(θ)

Helper function for plotting
"""
function βtitle(θ::MBreward)
    return "β(MBreward)"
end

"""
    agent2string(θ)

Gets string corresponding to agent
"""
function agent2string(θ::MBreward)
    return "MBreward"
end

function atick(θ::MBreward)
    return "MBr"
end