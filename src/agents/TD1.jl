"""
    TD1{T} <: Agent

Parameters for an agent computing values as a model-free (TD1) agent. (Exactly the same as θMFreward())

Fields:
- `Qinit`: initial Q values
- `α`: learning rate. randomly initialized by default from a Beta distribution
- `θL2`: standard deviation of normal distribution to initialize GLM weights
- `color`: color of agent for plotting
- `color_lite`: lighter color of agent for plotting
"""
@with_kw struct TD1{T<:Real,Q<:AbstractVector,I<:Int,A<:Distribution,B<:AbstractVector,C<:Symbol} <: Agent 
    Q0::Q = @SVector [0.5,0.5,0,0]
    α::T = rand(Beta(5,5))
    αprior::A = Beta(2,2)
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
function next_Q!(Q::AbstractArray{T},θ::TD1,data::RatData,t::Int) where T <: Union{Float64,ForwardDiff.Dual}
    @unpack α = θ
    @unpack choices,rewards = data
    Q .*= (1 - α)
    Q[choices[t]] += α*rewards[t]
end

function αtitle(θ::TD1)
    return "α(TD1)"
end


function βtitle(θ::TD1)
    return "β(TD1)"
end

function agent2string(θ::TD1)
    return "TD1"
end