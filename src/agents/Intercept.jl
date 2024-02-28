"""
    Intercept{Q,L,C} <: Agent where {Q<:AbstractVector,L<:Float64,C<:Union{Symbol,Vector{Float64}}}

Parameters for an agent defining a fixed intercept across trials
(value difference = 1, half of Bias agent)

Fields:
- `Q0::Q`: initial Q values
- `θL2::L`: standard deviation of normal distribution to initialize GLM weights; also used for L2 penalty
- `color::C`: color of agent for plotting
- `color_lite::C`: lighter color of agent for plotting
"""
@with_kw struct Intercept{Q<:AbstractVector,B<:AbstractVector,C<:Symbol} <: Agent 
    Q0::Q = @SVector [0.5,-0.5,0.,0.]
    βprior::B = [Normal(0,10)]
    color::C = :black
    color_lite::C = :gray
end

function agent_color(agent::Intercept)
    return agent.color_lite
end

"""
    next_Q!(Q::AbstractArray{Float64},θ::Intercept,data::D,t::Int) where D <: RatData

Updates the Q value (in-place) given its previous value, agent parameters, and trial data.

Arguments:
- `Q`: previous Q value
- `θ`: agent parameters
- `data`: behavioral data
- `t`: trial number to pull from data
"""
function next_Q!(Q::AbstractArray,θ::Intercept,data::D,t::Int) where D <: RatData
    Q[1] = 1.
    Q[2] = -1.
end

"""
    βtitle(θ)

Helper function for plotting
"""
function βtitle(θ::Intercept)
    return "β(Intercept)"
end

function atick(θ::Intercept)
    return "Intercept"
end

"""
    agent2string(θ)

Gets string corresponding to agent
"""
function agent2string(θ::Intercept)
    return "Intercept"
end