"""
    MBchoiceB{T} <: Agent

Parameters for an agent computing values as a model-based choice kernel from an Ito & Doya model-based reinforcement learning framework. Unlike θMBchoice, the learning rate `α` on choice is absorbed into the agent weight `β` to reduce interactions between the two parameters.

Fields:
- `Qinit`: initial Q values
- `α`: learning rate. randomly initialized by default from a Beta distribution
- `θL2`: standard deviation of normal distribution to initialize GLM weights
- `color`: color of agent for plotting
- `color_lite`: lighter color of agent for plotting
"""
@with_kw struct MBchoiceB{T<:Real,Q<:AbstractVector,I<:Int,B<:AbstractVector,A<:Distribution,C<:Symbol} <: Agent 
    Q0::Q = @SVector [0.5,0.5,0,0]
    α::T = rand(Beta(5,5))
    αprior::A = Beta(1,1)
    αscale::I = 1
    βprior::B = [Normal(0,10)]
    color::C = :slateblue4
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
function next_Q!(Q::AbstractArray,θ::MBchoiceB,data::TwoStepData,t::Int) 
    @unpack α = θ
    @unpack choices,nonchoices,trans_commons = data
    Q .*= (1 - α)
    if trans_commons[t]
        Q[choices[t]] += 1
    else
        Q[nonchoices[t]] += 1
    end
end

"""
    αtitle(θ)

Helper function for plotting
"""
function αtitle(θ::MBchoiceB)
    return "α(MBc)"
end

"""
    βtitle(θ)

Helper function for plotting
"""
function βtitle(θ::MBchoiceB)
    return "β(MBc)"
end

function atick(θ::MBchoiceB)
    return "MBc"
end

"""
    agent2string(θ)

Gets string corresponding to agent
"""
function agent2string(θ::MBchoiceB)
    return "MBchoiceB"
end
