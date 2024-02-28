"""
    Persev{T} <: Agent

Parameters for an agent tracking 1-trial-back perseveration (model-free perseveration)

Fields:
- `Qinit`: initial Q values
- `θL2`: standard deviation of normal distribution to initialize GLM weights; also used for L2 regularization
- `color`: color of agent for plotting
- `color_lite`: lighter color of agent for plotting
"""
@with_kw struct Persev{Q<:AbstractVector,B<:AbstractArray,C<:Symbol} <: Agent 
    Q0::Q = @SVector [0.5,0.5,0,0]
    βprior::B = [Normal(0,10)]
    color::C = :steelblue
    color_lite::C = :lightblue
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
function next_Q!(Q::AbstractArray{Float64},θ::Persev,data::R,t::Int) where R <: RatData
    @unpack choices = data
    Q[1:2] .= 0
    Q[choices[t]] = 1
end

"""
    αtitle(θ)

Helper function for plotting
"""
function βtitle(θ::Persev)
    return "β(Persev)"
end

"""
    agent2string(θ)

Gets string corresponding to agent
"""
function agent2string(θ::Persev)
    return "Persev"
end