"""
    θMBreward{T} <: Agent

Parameters for an agent computing values using a model-based temporal difference learning rule based on Bellman's equation

Fields:
- `Qinit`: initial Q values
- `α`: learning rate. randomly initialized by default from a Beta distribution
- `θL2`: standard deviation of normal distribution to initialize GLM weights; also used for L2 penalty
- `color`: color of agent for plotting
- `color_lite`: lighter color of agent for plotting
"""
@with_kw struct MBbellman{T<:Real,Q<:AbstractVector,I<:Int,B<:AbstractArray,A<:Distribution,C<:Symbol} <: Agent 
    Q0::Q = @SVector [0.5,0.5,0.5,0.5]
    α::T = rand(Beta(5,5))
    βprior::B = [Normal(0,10)]
    αprior::A = Beta(1,1)
    αscale::I = 1
    color::C = :purple
    color_lite::C = :lavender
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
function next_Q!(Q::AbstractArray,θ::MBbellman,data::TwoStepData,t::Int) 
    @unpack α = θ
    @unpack outcomes,rewards,p_congruent = data
    Q .*= (1 - α)
    Q[2+outcomes[t]] += α*rewards[t]
    Q[1] = p_congruent*Q[3] + (1 - p_congruent)*Q[4]
    Q[2] = p_congruent*Q[4] + (1 - p_congruent)*Q[3]
end

"""
    αtitle(θ)

Helper function for plotting
"""
function αtitle(θ::MBbellman)
    return "α(MB)"
end

"""
    βtitle(θ)

Helper function for plotting
"""
function βtitle(θ::MBbellman)
    return "β(MB)"
end

"""
    agent2string(θ)

Gets string corresponding to agent
"""
function agent2string(θ::MBbellman)
    return "MBbellman"
end