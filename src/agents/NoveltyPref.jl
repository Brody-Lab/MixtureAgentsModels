"""
    NoveltyPref{T} <: Agent

Parameters for an agent tracking 1-trial-back novelty preference (model-based perseveration)

Fields:
- `Qinit`: initial Q values
- `θL2`: standard deviation of normal distribution to initialize GLM weights; also used for L2 regularization
- `color`: color of agent for plotting
- `color_lite`: lighter color of agent for plotting
"""
@with_kw struct NoveltyPref{Q<:AbstractVector,B<:AbstractVector,C<:Symbol} <: Agent 
    Q0::Q = @SVector [0.5,0.5,0.,0.]
    βprior::B = [Normal(0,10)]
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
function next_Q!(Q::AbstractArray,θ::NoveltyPref,data::TwoStepData,t::Int)
    @unpack choices,trans_commons = data
    if trans_commons[t]
        Q[1:2] .= 0.
        Q[choices[t]] = 1.
    else
        Q[1:2] .= 1.
        Q[choices[t]] = 0.
    end
end

"""
    βtitle(θ)

Helper function for plotting
"""
function βtitle(θ::NoveltyPref)
    return "β(NP)"
end

"""
    agent2string(θ)

Gets string corresponding to agent
"""
function agent2string(θ::NoveltyPref)
    return "NP"
end