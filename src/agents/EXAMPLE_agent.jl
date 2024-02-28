"""
    EXAMPLEagent{T1<:Real,T2<:Real,D1<:Distribution,D2<:Distribution,Q<:AbstractVector,I<:Int,B<:AbstractArray,C<:Symbol} <: Agent 

Example framework with required fields and functions for making a new agent type

Per julia guidelines, it's recommended to explicitly define the type of each field in the constructor. This ensures faster compilation times and faster performance. Have separately defined types for parameters you're interested in fitting in the case that you only fit one / keep the other fixed.

Required fields:
- `Q0`: initial Q values with length of four. 1=primary and 2=secondary choices. 3 and 4 can correspond to additional state values (e.g. outcome value) used in update rule. Set to `0` if they are not needed. Required for concatenation compatibility with agents requiring four states (for an example, see `MBbellman.jl`)
- `βprior`: vector of prior distributions / L2 penalty
- `color`: color of agent for plotting
- `color_lite`: lighter color of agent for plotting

Optional fields:
- `PARAM1`: some parameter you want to fit, e.g. a learning rate. must include `scale`; `prior` only necessary if you want to fit it
- `PARAM1scale`: scale for parameter, 0 = no scaling, 1 = scaled using logistic (to ensure values between 0 and 1 for unbounded fit)
- `PARAM1prior`: some prior distribution. Set to `Normal(0,Inf)` for no prior, or `Beta(1,1)` for no prior on parameter that scales between 0 and 1
- `PARAM2`: another parameter you want to fit, e.g. a discount factor
- `PARAM2scale`
- `PARAM2prior`
"""
@with_kw struct EXAMPLEagent{T1<:Real,T2<:Real,D1<:Distribution,D2<:Distribution,Q<:AbstractVector,I<:Int,B<:AbstractArray,C<:Symbol} <: Agent 
    Q0::Q = @SVector [0.5,0.5,0,0]
    PARAM1::T1 = rand(Beta(5,5))  # or whatever distribution you want to use for random initialization
    PARAM1scale::I = 1  # scale for PARAM1, 0 = no scaling, 1 = scaled using logistic (to ensure values between 0 and 1)
    PARAM1prior::D1 = Beta(1,1)  # prior distribution for PARAM1
    PARAM2::T2 = rand(Beta(5,5))
    PARAM2scale::I = 1
    PARAM2prior::D2 = Beta(1,1)
    βprior::B = [Normal(0,10)]  # prior distribution for β. make sure it's in a vector (length > 1 if you want different priors for different HMM states)
    color::C = :purple
    color_lite::C = :lavender
end

"""
    next_Q!(Q::Array{Float64},θ::EXAMPLEagent,data::D,t::Int) where D <: RatData

Updates the Q value given its previous value, agent parameters, and trial data.

Arguments:
- `Q`: previous Q value
- `θ`: agent parameters
- `data`: behavioral data
- `t`: trial number to pull from data
"""
function next_Q!(Q::T,θ::EXAMPLEagent,data::D,t::Int) where {T <: AbstractArray, D <: RatData}
    @unpack PARAM1,PARAM2 = θ  # unpack parameters used for updating values, if any exist; otherwise remove this line
    @unpack choices,rewards = data # unpack behavioral data used for updating values, if any are needed; otherwise remove this line

    # update Q values according to whatever rule you want. 
    # for example, if YOUR_PARAM is a learning rate, a TD1 update would look like:
    Q .*= (1 - PARAM1) # use broadcasting to ensure in-place assignment when modifying all of Q
    Q[choices[t]] += PARAM1*rewards[t]  # trial t indexes choices, giving 1 or 2, which indexes Q
end

"""
    αtitle(θ)

Helper function for plotting, if YOUR_PARAM is a learning rate
"""
function αtitle(θ::EXAMPLEagent)
    return "α(EXAMPLE)"
end

"""
    βtitle(θ)

Helper function for plotting
"""
function βtitle(θ::EXAMPLEagent)
    return "β(MB)"
end

"""
    agent2string(θ)

Gets string corresponding to agent, useful for saving
"""
function agent2string(θ::EXAMPLEagent)
    return "EXAMPLE"
end