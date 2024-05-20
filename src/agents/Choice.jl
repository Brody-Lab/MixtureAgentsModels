"""
    Choice <: Agent

Parameters for an agent capturing effect of previous choice made at a specified trial lag.

Fields:
- `Qinit`: initial Q values
- `nback`: number of trials to look back
- `θL2`: standard deviation of normal distribution to initialize GLM weights
- `color`: color of agent for plotting
- `color_lite`: lighter color of agent for plotting
"""
@with_kw struct Choice{I<:Int,Q<:AbstractVector,B<:AbstractVector,C<:Symbol,S<:Symbol} <: Agent
    Q0::Q = @SVector [0.5,0.5,0,0]
    nback::I = 1; @assert nback > 0
    βprior::B = [Normal(0,10)]
    color::C = :royalblue
    color_lite::C = :salmon
    line_style::S = :solid
end

"""
    Choice(nback)

Helper function to create a `Choice` agent with the specified number of trials to look back. 
If `nback` is an array, it will return a vector of `Choice` agents corresponding to each element of `nback`.
"""
function Choice(nback::Int)
    return Choice(nback=nback)
end
function Choice(nback::AbstractArray{Int})
    return vcat(Choice.(nback)...)
end

"""
    next_Q(Q, θ, data, t)

Updates the Q value given its previous value, agent parameters, and trial data.

Arguments:
- `Q`: previous Q value
- `θ`: agent parameters
- `data`: behavioral data
- `t`: index of last trial
"""
function next_Q!(Q::AbstractArray{Float64},θ::A,data::D,t::Int) where {A <: Choice, D <: RatData}
    @unpack choices,new_sess = data
    @unpack nback = θ
    tn = t-nback+1
    Q .= 0.
    if tn > 0
        if !any(new_sess[tn+2:t])
            Q[choices[tn]] = 1.
        end
    end
end

"""
    βtitle(θ)

Helper function for plotting
"""
function βtitle(θ::Choice)
    @unpack nback = θ
    return string("β(Choice[t-",nback,"])")
end

"""
    agent2string(θ)

Gets string corresponding to agent
"""
function agent2string(θ::Choice)
    @unpack nback = θ
    return string("Choice",nback)
end