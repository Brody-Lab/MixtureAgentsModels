"""
    θΔClicks{Q<:AbstractVector,B<:AbstractVector,C<:Symbol} <: Agent 

Agent corresponding to a simple delta rule for right and left clicks on the Poisson clicks task.
Assumes right choices as 1 and left choices as 2.

Fields:
- `Q0`: initial Q values with length of four. 1=right and 2=left choices. 3 and 4 can correspond to additional state values (e.g. outcome value) used in update rule. Set to `0` if they are not needed. Required for concatenation compatibility with agents requiring four states (for an example, see `MBbellman.jl`)
- `βprior`: vector of prior distributions / L2 penalty
- `color`: color of agent for plotting
- `color_lite`: lighter color of agent for plotting

"""
@with_kw struct DeltaClicks{Q<:AbstractVector,B<:AbstractVector,C<:Symbol} <: Agent 
    Q0::Q = @SVector [0,0,0,0]
    βprior::B = [Normal(0,10)]  # prior distribution for β. make sure it's in a vector (length > 1 if you want different priors for different HMM states)
    color::C = :purple
    color_lite::C = :lavender
end

"""
    init_Q(θ::DeltaClicks; data::D=nothing, t::Int=1) where D <: Union{PClicksData,Nothing}

Special initialization function for DeltaClicks agent using the clicks on the first trial, since Q values are updated after the first trial.
Uses default initialization if no data is provided.
"""
function init_Q(θ::DeltaClicks; data::D=nothing, t::Int=1) where D <: Union{PClicksData,Nothing}
    if !isnothing(data)
        @unpack nleftclicks,nrightclicks = data
        return @SVector [nrightclicks[t],nleftclicks[t],0,0]
    else
        return θ.Q0
    end
end

"""
    next_Q!(Q::T,θ::DeltaClicks,data::D,t::Int) where {T <: AbstractArray, D <: RatData}

Updates the Q value to the difference in right and left clicks on the next trial.

Arguments:
- `Q`: Container for Q value update
- `θ`: agent parameters
- `data`: behavioral data
- `t`: trial number to pull from data
"""
function next_Q!(Q::T,θ::DeltaClicks,data::D,t::Int) where {T <: AbstractArray, D <: RatData}
    @unpack nleftclicks,nrightclicks = data 

    if t < length(nrightclicks)
        Q[1] = nrightclicks[t+1]  
        Q[2] = nleftclicks[t+1]
    else
        Q[1] = 0
        Q[2] = 0
    end
end


"""
    βtitle(θ)

Helper function for plotting
"""
function βtitle(θ::DeltaClicks)
    return "β(ΔClicks)"
end

"""
    agent2string(θ)

Gets string corresponding to agent, useful for saving
"""
function agent2string(θ::DeltaClicks)
    return "DeltaClicks"
end


function atick(θ::DeltaClicks)
    return "ΔClicks"
end
