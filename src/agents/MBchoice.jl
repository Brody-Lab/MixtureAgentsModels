"""
    θMBchoice{T} <: Agent

Parameters for an agent computing values as a model-based choice kernel from an
Ito & Doya model-based reinforcement learning framework

Fields:
- `Qinit`: initial Q values
- `α`: learning rate. randomly initialized by default from a Beta distribution
- `θL2`: standard deviation of normal distribution to initialize GLM weights
- `color`: color of agent for plotting
- `color_lite`: lighter color of agent for plotting
"""
@with_kw struct MBchoice{T<:Real,Q<:AbstractVector,I<:Int,A<:Distribution,B<:AbstractVector,C<:Symbol} <: Agent 
    Q0::Q = @SVector [0.5,0.5,0,0]
    α::T = rand(Beta(5,5))
    αprior::A = Beta(1,1)
    βprior::B = [Normal(0,10)]
    αscale::I = 1
    color::C = :maroon
    color_lite::C = :pink
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
function next_Q!(Q::AbstractArray,θ::MBchoice,data::TwoStepData,t::Int) 
    @unpack α = θ
    @unpack choices,nonchoices,trans_commons = data
    Q .*= (1 - α)
    if trans_commons[t]
        Q[choices[t]] += α
    else
        Q[nonchoices[t]] += α
    end
end

"""
    αtitle(θ)

Helper function for plotting
"""
function αtitle(θ::MBchoice)
    return "α(MBchoice)"
end

"""
    βtitle(θ)

Helper function for plotting
"""
function βtitle(θ::MBchoice)
    return "β(MBchoice)"
end

"""
    agent2string(θ)

Gets string corresponding to agent
"""
function agent2string(θ::MBchoice)
    return "MBchoice"
end

function atick(θ::MBchoice)
    return "MBc"
end
