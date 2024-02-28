"""
    TransReward <: Agent

Parameters for an agent capturing effect of transition x reward at defined trial lag. Reward x transition pairs are split into different structs. While I could combine them into a single struct, this allows the agent names to be more descriptive and require less parameters to initialize

Fields:
- `Qinit`: initial Q values
- `nback`: number of trials to look back
- `rew`: reward value to look for
- `tran`: transition type to look for
- `θL2`: standard deviation of normal distribution to initialize GLM weights
- `color`: color of agent for plotting
- `color_lite`: lighter color of agent for plotting
"""
abstract type TransReward <: Agent end

"""
    CR <: TransReward

1 for common-rewards, 0 otherwise
"""
@with_kw struct CR{I<:Int,Q<:AbstractVector,B<:AbstractVector,C<:Symbol,S<:Symbol} <: TransReward 
    Q0::Q = @SVector [0.5,0.5,0,0]
    nback::I = 1; @assert nback > 0
    rew::I = 1; @assert rew == 1
    tran::I = 1; @assert tran == 1
    βprior::B = [Normal(0,10)]
    color::C = :royalblue
    color_lite::C = :salmon
    line_style::S = :solid
end
function CR(nback)
    return CR(nback=nback)
end

"""
    CO <: TransReward

1 for common-omissions, 0 otherwise
"""
@with_kw struct CO{I<:Int,Q<:AbstractVector,B<:AbstractVector,C<:Symbol,S<:Symbol} <: TransReward 
    Q0::Q = @SVector [0.5,0.5,0,0]
    nback::I = 1; @assert nback > 0
    rew::I = -1
    tran::I = 1
    βprior::B = [Normal(0,10)]
    color::C = :firebrick
    color_lite::C = :steelblue1
    line_style::S = :solid
end
function CO(nback)
    return CO(nback=nback)
end

"""
    UR <: TransReward

1 for uncommon-rewards, 0 otherwise
"""
@with_kw struct UR{I<:Int,Q<:AbstractVector,B<:AbstractVector,C<:Symbol,S<:Symbol} <: TransReward 
    Q0::Q = @SVector [0.5,0.5,0,0]
    nback::I = 1; @assert nback > 0
    rew::I = 1
    tran::I = 0
    βprior::B = [Normal(0,10)]
    # color::Union{Symbol,Vector{Float64}} = :goldenrod2
    # color_lite::Union{Symbol,Vector{Float64}} = :lightgoldenrod1
    color::C = :royalblue
    color_lite::C = :salmon
    line_style::S = :dash
end
function UR(nback)
    return UR(nback=nback)
end

"""
    UO <: TransReward

1 for uncommon-omissions, 0 otherwise
"""
@with_kw struct UO{I<:Int,Q<:AbstractVector,B<:AbstractVector,C<:Symbol,S<:Symbol} <: TransReward 
    Q0::Q = @SVector [0.5,0.5,0,0]
    nback::I = 1; @assert nback > 0
    rew::I = -1
    tran::I = 0
    βprior::B = [Normal(0,10)]
    # color::Union{Symbol,Vector{Float64}} = :seagreen3
    # color_lite::Union{Symbol,Vector{Float64}} = :palegreen
    color::C = :firebrick
    color_lite::C= :steelblue1
    line_style::S = :dash
end
function UO(nback)
    return UO(nback=nback)
end

"""
    TransReward(nback)

Helper function to initialize all TransReward structs given an nback integer or integer array
"""
function TransReward(nback::Int)
    return [CR(nback),UR(nback),CO(nback),UO(nback)]
end
function TransReward(nback::AbstractArray{Int})
    return vcat(TransReward.(nback)...)
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
function next_Q!(Q::AbstractArray{Float64},θ::A,data::D,t::Int) where {A <: TransReward, D <: RatData}
    @unpack choices,rewards,trans_commons,new_sess = data
    @unpack nback,rew,tran = θ
    tn = t-nback+1
    Q .= 0.
    if tn > 0
        if !any(new_sess[tn+2:t])
            Q[choices[tn]] = Float64((rewards[tn]==rew) && (trans_commons[tn]==tran))
        end
    end
end

"""
    βtitle(θ)

Helper function for plotting
"""
function βtitle(θ::CR)
    @unpack nback = θ
    return string("β(CR-",nback,")")
end
function βtitle(θ::CO)
    @unpack nback = θ
    return string("β(CO-",nback,")")
end
function βtitle(θ::UR)
    @unpack nback = θ
    return string("β(UR-",nback,")")
end
function βtitle(θ::UO)
    @unpack nback = θ
    return string("β(UO-",nback,")")
end

"""
    agent2string(θ)

Gets string corresponding to agent
"""
function agent2string(θ::CR)
    @unpack nback = θ
    return string("CR",nback)
end
function agent2string(θ::CO)
    @unpack nback = θ
    return string("CO",nback)
end
function agent2string(θ::UR)
    @unpack nback = θ
    return string("UR",nback)
end
function agent2string(θ::UO)
    @unpack nback = θ
    return string("UO",nback)
end