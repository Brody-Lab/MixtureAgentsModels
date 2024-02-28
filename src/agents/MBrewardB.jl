"""
    MBrewardB{T} <: Agent

Parameters for an agent computing values as a model-based reward kernel from an Ito & Doya model-based reinforcement learning framework. Unlike θMBreward, the learning rate `α` on reward is absorbed into the agent weight `β` to reduce interactions between the two parameters.

Fields:
- `Qinit`: initial Q values
- `α`: learning rate. randomly initialized by default from a Beta distribution
- `θL2`: standard deviation of normal distribution to initialize GLM weights
- `color`: color of agent for plotting
- `color_lite`: lighter color of agent for plotting
"""
@with_kw struct MBrewardB{T<:Real,Q<:AbstractVector,I<:Int,B<:AbstractVector,A<:Distribution,C<:Symbol} <: Agent
    Q0::Q = @SVector [0.5,0.5,0,0]
    α::T = rand(Beta(5,5))
    βprior::B = [Normal(0,10)]
    αprior::A = Beta(1,1)
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
function next_Q!(Q::AbstractArray{T},θ::MBrewardB,data::TwoStepData,t::Int) where T <: Union{Float64,ForwardDiff.Dual}
    @unpack α = θ
    @unpack choices,nonchoices,rewards,trans_commons = data
    Q .*= (1 - α)
    if trans_commons[t]
        Q[choices[t]] += rewards[t]
    else
        Q[nonchoices[t]] += rewards[t]
    end
end

"""
    αtitle(θ)

Helper function for plotting
"""
function αtitle(θ::MBrewardB)
    return "α(MBr)"
end

"""
    βtitle(θ)

Helper function for plotting
"""
function βtitle(θ::MBrewardB)
    return "β(MBr)"
end

function atick(θ::MBrewardB)
    return "MBr"
end



"""
    agent2string(θ)

Gets string corresponding to agent
"""
function agent2string(θ::MBrewardB)
    return "MBrewardB"
end
