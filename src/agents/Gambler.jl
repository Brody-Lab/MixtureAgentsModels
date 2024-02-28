"""
    Gambler{T} <: Agent

Parameters for an agent computing values from the perspective of a Gambler's Fallacy

Fields:
- `Qinit`: initial Q values
- `α`: learning rate. randomly initialized by default from a Beta distribution
- `color`: color of agent for plotting
- `color_lite`: lighter color of agent for plotting
"""

@with_kw struct Gambler{T<:Real,Q<:AbstractVector,I<:Int,A<:Distribution,B<:AbstractVector,C<:Symbol} <: Agent
    Q0::Q = @SVector [0.,0.,0.,0.]
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
function next_Q!(Q::AbstractArray,θ::Gambler,data::D,t::Int) where D <: RatData
    @unpack α = θ
    @unpack choices,rewards = data
    Q .*= (1 - α)
    Q[choices[t]] += α*(1 - rewards[t])
end

"""
    αtitle(θ)

Helper function for plotting
"""
function αtitle(θ::Gambler)
    return "α(Gambler)"
end

"""
    βtitle(θ)

Helper function for plotting
"""
function βtitle(θ::Gambler)
    return "β(Gambler)"
end

"""
    agent2string(θ)

Gets string corresponding to agent
"""
function agent2string(θ::Gambler)
    return "Gambler"
end

function atick(θ::Gambler)
    return "GF"
end
