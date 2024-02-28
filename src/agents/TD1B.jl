"""
    TD1B{T} <: Agent

Parameters for an agent computing values as a model-free (TD1) agent. Unlike θTD1, the learning rate on reward `α` is absorbed into the agent weight `β` to reduce interactions between the two parameters.

Fields:
- `Qinit`: initial Q values
- `α`: learning rate. randomly initialized by default from a Beta distribution
- `θL2`: standard deviation of normal distribution to initialize GLM weights
- `color`: color of agent for plotting
- `color_lite`: lighter color of agent for plotting
"""
@with_kw struct TD1B{T<:Real,Q<:AbstractVector,I<:Int,A<:Distribution,B<:AbstractVector,C<:Symbol} <: Agent 
    Q0::Q = @SVector [0.5,0.5,0,0]
    α::T = rand(Beta(5,5))
    αprior::A = Beta(1,1)
    βprior::B = [Normal(0,10)]
    αscale::I = 1
    color::C = :gold3
    color_lite::C = :gold3
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
function next_Q!(Q::AbstractArray,θ::TD1B,data::RatData,t::Int) 
    @unpack α = θ
    @unpack choices,rewards = data
    Q .*= (1 - α)
    Q[choices[t]] += rewards[t]
end


function αtitle(θ::TD1B)
    return "α(MF)"
end


function βtitle(θ::TD1B)
    return "β(MF)"
end

function agent2string(θ::TD1B)
    return "TD1B"
end

function atick(θ::TD1B)
    return "MF"
end