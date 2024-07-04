"""
    AgentOptions{A<:Agent,VS<:Union{Vector{Symbol},Nothing},VI<:Union{Vector{Int},Nothing},VP<:Union{Vector{Symbol},Nothing},B<:Bool} 

Options indicating agents and agent parameters that need to get fit. See constructor function for usage. Fields here are seemingly redudant to reduce computation time when fitting multiple agents that share parameters.

# Fields:
- `agents`: (nagents x 1) or (nagents x nstates) Array of agents to be fit by model
- `fit_symbs`: (nparams x 1) Vector of symbols corresponding to each agent parameter to fit.
- `symb_inds`: (nparams x 1) (agent_inds) Vector specifying which agent to pull the parameter from. Calculated automatically when using the constructor function
- `fit_params`: (nfit x 1) Flattenned vector of symbol indices. Non-fit agents (i.e. agents with index `0`) are removed. Converted automatically when using the constructor function
- `param_inds`: (nfit x 1) (agents_fit) Flattened vector of agent indices paired with symbol indices from `fit_params`. Calculated automatically when using the constructor function
- `fit_inds`: (nfit_agents x 1) Vector specify which agents are fit. Calculated automatically from constructor function
- `fit_scales`: (nparams x 1) Vector of symbols for scaling parameters of each fit parameter. Calculated automatically from constructor function.
- `fit_priors`: (nparams x 1) Vector of symbols for scaling parameters of each fit parameter. Caluclated in constructor function if `fit_priors=true`
- `scale_x`: (Bool) Whether to scale (by zscoring) the value difference matrix `x`. Defaults to `false`
"""
@with_kw struct AgentOptions{A<:Agent,VS<:Union{Vector{Symbol},Nothing},VI<:Union{Vector{Int},Nothing},VP<:Union{Vector{Symbol},Nothing},B<:Bool} 
    agents::Array{A}
    fit_symbs::VS = nothing
    symb_inds::VI = nothing
    fit_params::VI = nothing
    param_inds::VI = nothing
    fit_inds::VI = nothing
    fit_scales::VS = nothing
    fit_priors::VP = nothing
    scale_x::B = false
end


"""
    AgentOptions(agents;scale_x=false)
    AgentOptions(agents,fit_symbs,fit_params;fit_priors=false,scale_x=false)

Constructor function to define `AgentOptions` struct. Converts inputs and caculates additional fields as needed.

# Inputs:
- `agents`: (nagents x 1) or (nagents x nstates) Array of agents to be fit by model.
- `fit_symbs`: (nparams x 1) Vector of symbols corresponding to each agent parameter to fit. Symbol must match field in appropriate agent struct
- `fit_params`: (nagent x 1) Vector of (integers, vector of integers, integer range) linking each agent with associated parameter in `fit_symbs`. See example for use cases.

# Optional Inputs:
- `fit_priors`: (Bool) Whether to use priors for each fit parameter. Defaults to `false`
- `scale_x`: (Bool) Whether to scale (by zscoring) the value difference matrix `x`. Defaults to `false`

# Examples:
## Example 1: Separate parameters
Specify 5 agents defined by MBrewardB.jl, MBchoiceB.jl, MFrewardB.jl, MFchoiceB.jl, and Bias.jl. 

    julia> agents = [θMBrewardβ(),θMBchoiceβ(),θMFrewardβ(),θMFchoiceβ(),θBias()]

Specify four learning rates each with symbol `:α` to fit, one for each RL agent

    julia> fit_symbs = [:α,:α,:α,:α]

Index for each agent corresponding to `fit_symbs`. `θBias()` does not have any parameters being fit. 

    julia> fit_params = [1,2,3,4,0]
    julia> options = agentsoptions(agents,fit_symbs,fit_params)

## Example 2: Shared parameters
Fit 5 agents defined by MBreward.jl, MBchoice.jl, MFreward.jl, MFchoice.jl, and Bias.jl.

    julia> agents = [θMBreward(),θMBchoice(),θMFreward(),θMFchoice(),θBias()]

Fit two parameters both with symbol `:α`

    julia> fit_symbs = [:α,:α]

The first `α` is shared between MBreward and MBchoice, and the second `α` shared between MFreward and MFchoice. Bias does not have any parameters being fit.
    
    julia> fit_params = [1,1,2,2,0]
    julia> options = agentsoptions(agents,fit_symbs,fit_params)

## Example 3: Multiple parameters per agent
Fit 3 agents defined by TD1.jl, TD0.jl, and Bias.jl. 

    julia> agents = [θTD1(),θTD0(),θBias()]

Fit two parameters, one with symbol `:α` and the other with symbol `:γ`
    
    julia> fit_symbs = [:α,:γ]

`α` is shared between TD1 and TD0. `γ` is only fit for TD0. No parameters for Bias

    julia> fit_params = [1, 1:2, 0]   # or equivalently = [1, [1,2], 0]
    julia> options = agentsoptions(agents,fit_symbs,fit_params)

## Example 4: No fit parameters (equivalent to a GLM-HMM)
Fit 3 agents defined by NP.jl, Persev.jl, and Bias.jl. 

    julia> agents = [θNP(),θPersev(),θBias()]

No agent parameters to fit. Define options by passing only the agents, which will utilyze default values of `nothing` for `fit_symbs` and `fit_params`
    
    julia> options = AgentOptions(agents)
"""
function AgentOptions(agents::Array{A},fit_symbs::Vector{Symbol},fit_params::Vector{T};fit_priors::Bool=false,scale_x::Bool=false) where {A <: Agent, T <: Any}
    # AgentOptions constructor for flattening `fit_params` when an agent has multiple parameters to fit
    fit_params_f = reduce(vcat,fit_params) # flattened fit_params
    fit_params_r = copy(fit_params) # copy of original fit_params in case of repeat manipulation
    fit_symbs_r = copy(fit_symbs) # copy of original fit_symbs in case of repeat manipulation
    fit_scales_r = [Symbol(string(s)*"scale") for s in fit_symbs_r] # grab scale symbol for each fit parameter
    if fit_priors
        fit_priors_r = [Symbol(string(s)*"prior") for s in fit_symbs_r] # grab prior symbol for each fit parameter
    else
        fit_priors_r = nothing
    end
    if length(fit_params) < length(agents) # for agents dependent on latent state when parameters are specified for only 1 state and are repeated
        ns = size(agents,2) # number of states
        np = length(fit_symbs) # number of params
        na = length(fit_params_f)   
        fit_symbs_r = repeat(fit_symbs_r,ns) # "flattened" nparams x nstates fit_symbs
        fit_scales_r = repeat(fit_scales_r,ns) # "flattened" nparams x nstates fit_scales
        if fit_priors
            fit_priors_r = repeat(fit_priors_r,ns)
        end
        fit_params_f = repeat(fit_params_f,ns) # "flattened" nagents x nstates fit_params
        fit_params_f[fit_params_f.>0] += vcat([zeros(Int,na) .+ x*np for x=0:ns-1]...)[fit_params_f.>0] # corrected index for flattened params
        fit_params_r = repeat(fit_params_r,ns)
    end
    param_inds = get_param_inds(fit_params_r) #reduce(vcat,[fill(i,axes(fit_params_r[i])) for i in eachindex(fit_params_r)]) # agent indices for fit_params
    return AgentOptions(agents,fit_symbs_r,fit_params_f,param_inds,fit_scales_r,fit_priors_r,scale_x)
end

function AgentOptions(agents::Array{A},fit_symbs::Symbol,fit_params::String;kwargs...) where {A <: Agent}
    # catch constructor for `dict2agents` conversion function when `fit_params` gets read as a string = "nothing"
    return AgentOptions(agents;kwargs...)
end

function AgentOptions(agents::Array{A},fit_symbs::Vector{Symbol},fit_params::Vector{Int64},param_inds::Vector{Int64},fit_scales::Vector{Symbol},fit_priors::Union{Vector{Symbol},Nothing},scale_x::Bool) where {A <: Agent}   
    # AgentOptions constructor function to add `fit_inds` and `symb_inds` fields, separated out for calls from `agent_comparison`.
    symb_inds = [param_inds[findfirst(fit_params .== a)] for a in unique(fit_params[fit_params .> 0])]
    deleteat!(param_inds,fit_params .== 0)
    deleteat!(fit_params,fit_params .== 0)
    fit_inds = unique(param_inds)
    return AgentOptions(agents=agents,fit_symbs=fit_symbs,symb_inds=symb_inds,fit_params=fit_params,param_inds=param_inds,fit_inds=fit_inds,fit_scales=fit_scales,fit_priors=fit_priors,scale_x=scale_x)
end

function AgentOptions(agents::Array{A};scale_x::Bool=false) where A <: Agent
    # AgentOptions constructor when no agent parameters are being fit
    return AgentOptions(agents=agents,scale_x=scale_x)
end

function AgentOptions(agents::Array{A},options::AgentOptions;fit_priors::Union{Bool,Nothing}=nothing) where A <: Agent
    # AgentOptions constructor to create stuct from existing AgentOptions and new agents with reinitialized parameter values.
    if isnothing(fit_priors) 
        if !isnothing(options.fit_priors)
            fit_priors = true
        else
            fit_priors = false
        end
    end
    return AgentOptions(agents,options.fit_symbs,get_fit_params(options);fit_priors=fit_priors,scale_x=options.scale_x)
end

function get_param_inds(fit_params::Vector{T}) where T
    # Compute `param_inds` from `fit_params`
    return reduce(vcat,[fill(i,axes(fit_params[i])) for i in eachindex(fit_params)])
end

function get_fit_params(options::AgentOptions)
    # Restores user-input-formatted `fit_params`
    @unpack agents, fit_params, param_inds = options
    if !isnothing(fit_params)
        na = length(agents)
        fit_params_tmp = Array{Any}(undef,na)
        for a = 1:na
            fit_params_tmp[a] = fit_params[param_inds .== a]
            if isempty(fit_params_tmp[a])
                fit_params_tmp[a] = 0
            end
        end
    else
        fit_params_tmp = "nothing"
    end
    return fit_params_tmp
end