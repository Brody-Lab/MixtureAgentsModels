"""
    agent_strings(agent::Union{AbstractString,Nothing}=nothing)

Dictionary matching all agent structs to a string key of its name. Agents with special characters in their name (e.g. `ΔClicks`) are additionally paried with a keystring with the special character replaced (e.g. `DeltaClicks`). Agents must be included in this dictionary to make use of built-in saving and loading functions.
"""
function agent_strings(agent::Union{AbstractString,Nothing}=nothing)
    agents = Dict(
        "Bias"=>Bias,
        "Intercept"=>Intercept,
        "MBbellman"=>MBbellman,
        "MBbellmanB"=>MBbellmanB,
        "MBchoice"=>MBchoice,
        "MBchoiceB"=>MBchoiceB,
        "MBreward"=>MBreward,
        "MBrewardB"=>MBrewardB,
        "MFchoice"=>MFchoice,
        "MFchoiceB"=>MFchoiceB,
        "MFreward"=>MFreward,
        "MFrewardB"=>MFrewardB,
        "NoveltyPref"=>NoveltyPref,"NP"=>NoveltyPref,
        "Persev"=>Persev,
        "TD0"=>TD0,
        "TD1"=>TD1,
        "TD1B"=>TD1B,
        "CR"=>CR,"CO"=>CO,"UR"=>UR,"UO"=>UO,
        "DeltaClicks"=>DeltaClicks,
        "DeltaClicksZ"=>DeltaClicksZ,
        "Gambler"=>Gambler)
    if !isnothing(agent)
        return agents[agent]
    else
        return agents
    end
end

"""
    optimize(model::M,agents::Array{A},data::D,model_ops::O,agent_ops::AgentOptions)

Optimizes model parameters using existing model and agents as starting point.
"""
function optimize(data::D,model::M,agents::Array{A},model_ops::O,agent_ops::AgentOptions) where {D <: RatData, A <: Agent, M <: MixtureAgentsModel, O <: ModelOptions}
    model_ops = Accessors.set(model_ops,opcompose(PropertyLens(:nstarts)),1) # force 1 start
    model_options,agent_options = fit2options(model,model_ops,agents,agent_ops)
    return optimize(data,model_options,agent_options;init_hypers=false)
end

"""
    agents_mean(agents::Union{Vector{Array{A}},Vector{Vector{A}},Vector{Matrix{A}}},options::AgentOptions) where A <: Agent

Function to compute mean parameter values across sets of agents. Returns new vector of agents with mean parameter values.
"""
function agents_mean(agents::Union{Vector{Array{A}},Vector{Vector{A}},Vector{Matrix{A}}},options::AgentOptions) where A <: Agent
    params_mean = mean(get_params.(agents,repeat([options],length(agents))))
    agents_new = deepcopy(agents[1])
    update!(agents_new,params_mean,options)
    return agents_new
end

"""
    initialize_x(agents::Array{A},data::D) where {A <: Agent, D <: RatData}

Initializes value difference matrix `x` and populates agent values for `agents` using `data`.
"""
function initialize_x(data::D,agents::Array{A}) where {A <: Agent, D <: RatData}
    @unpack nfree = data
    na = length(agents)
    x = Array{Float64}(undef,na,nfree)
    for a = 1:na
        x[a,:] = update!(x[a,:],agents[a],data)
    end
    return x
end

"""
    initialize_y(data::D) where D <: RatData

Initialize choice vector `y` using `data`. `1` corresponds to `choices = 1`; `0` corresponds to `choices = 2`
"""
function initialize_y(data::D) where D <: RatData
    @unpack choices,forced = data
    y = (choices .== 1)
    deleteat!(y,findall(forced))
    return y
end

"""
    initialize(data::D,agents::Array{A}) where {A <: Agent, D <: RatData}

Initializes value difference matrix `x` and choice vector `y` using `data`
"""
function initialize(data::D,agents::Array{A}) where {A <: Agent, D <: RatData}
    return initialize_y(data),initialize_x(data,agents)
end

"""
    initialize(options::AgentOptions)

Initializes `agents` by re-rolling fit parameters specified by `options`. Parameters not listed in `options` will be copied over
"""
function initialize(options::AgentOptions)
    @unpack agents,fit_params,symb_inds = options
    agents_new = deepcopy(agents)
    if !isnothing(fit_params)
        agents_param = initialize.(agents)
        params = get_params(agents_param,options)
        update!(agents_new,params,options)
    end
    return agents_new
end

"""
    initialize(agent::A) where {A <: Agent}

Initializes `agent`, setting all parameters to default values.
"""
function initialize(agent::A) where {A<:Agent}
    return A()
end

"""
    update!(agents::Array{A}, vals::Array{T}, options::AgentOptions) where {T <: Any, A <: Agent}

Update agents using new parameter values `vals` set according to `options`.
"""
function update!(agents::Array{A},vals::AbstractArray{T},options::AgentOptions) where {T <: Any, A <: Agent}
    @unpack fit_params,fit_symbs,param_inds = options
    if !isnothing(fit_params)
        for (a,i) in zip(param_inds,fit_params)
            agents[a] = update(agents[a],fit_symbs[i],vals[i])
        end
    end
end

"""
    update(agent::A, symbs::Array{Symbol}, vals::T) where {A <: Agent, T <: AbstractArray}

Wrapper for reconstructing agent with new parameter values listed in `symbs` and `vals`
"""
function update(agent::A,symbs::Array{Symbol},vals::T) where {A <: Agent, T <: AbstractArray}
    agent = Setfield.set(agent,IndexBatchLens(symbs...),vals)
    return agent
end

"""
    update(agent::A, symb::Symbol, val::T) where where {A <: Agent, T <: Real}

Reconstructs agent with new parameter value for `symb` in `val`.
"""
function update(agent::A,symb::Symbol,val::T) where {A <: Agent, T <: Any}
    agent = Accessors.set(agent,opcompose(PropertyLens(symb)),val)
    return agent
end

"""
    update!(x::Array{T1},agents::Array{A},vals::Array{T2},options::AgentOptions,data::D) where {A <: Agent, T1 <: Any, T2 <: Any, D <: RatData}

Populates value difference matrix `x` and updates agent vector `agents` with new parameter values `vals` set according to `options`.
Uses available threads to compute in parallel across agents.
"""
function update!(x::T1,agents::Array{A},vals::T2,options::AgentOptions,data::D) where {A <: Agent, T1 <: AbstractArray, T2 <: AbstractArray, D <: RatData}
    @unpack fit_params,fit_symbs,param_inds,fit_inds = options
    if !isnothing(fit_params)
        Threads.@threads for a in fit_inds
            i = fit_params[param_inds .== a]
            x[a,:],agents[a] = update(x[a,:],agents[a],vals[i],fit_symbs[i],data)
        end
    end
end

"""
    update(x::Array{T1},agent::A,vals::Union{T2,AbstractArray{T2}},symbs::Union{Symbol,AbstractArray{Symbol}},data::D) where {A <: Agent, T1 <: Any, T2 <: Any, D <: RatData}

Update `agent` parameters specified by `symbs` with new values `vals` and repopulated value difference matrix `x`
"""
function update(x::T1,agent::A,vals::Union{T2,AbstractArray{T2}},symbs::Union{Symbol,AbstractArray{Symbol}},data::D) where {A <: Agent, T1 <: AbstractArray, T2 <: Real, D <: RatData}
    agent = update(agent,symbs,vals)
    update!(x,agent,data)
    return x,agent
end

"""
    update!(x::Array{T},agent::A,data::D) where {A <: Agent, T <: Any, D <: RatData}

Populate value difference vector `x` for `agent` using `data`. Works on mutable variable `x`, but also returns `x` if input is immutable (e.g. when the passed `x` is an indexed subset of full value matrix). The value difference is computed as `Q[1] - Q[2]`, where the first index corresponds to the primary choice and the second index corresponds to the alternative choice.
Uses available threads to compute in parallel across sessions.
"""
function update!(x::Array{T},agent::A,data::D) where {A <: Agent, T <: Union{Float64,ForwardDiff.Dual}, D <: RatData}
    @unpack ntrials,new_sess,new_sess_free,sess_inds,forced = data
    Threads.@threads for (inds,tf) in collect(zip(sess_inds,findall(new_sess_free))) #eachindex(sess_inds) #nds in sess_inds
        Q = SizedVector{4}(init_Q(agent,T))
        for t in inds
            if new_sess[t]
                Q .= init_Q(agent;data=data,t=t)
            end
            if !(forced[t])
                x[tf] = Q[1] - Q[2]
                tf += 1
            end
            next_Q!(Q,agent,data,t)
        end
    end
    return x
end

"""
    init_Q(θ::Agent;T=Float64)

Returns initial Q values for agent `θ`
"""
function init_Q(θ::Agent; kwargs...)
    return copy(θ.Q0)
end

"""
    init_Q(θ::Agent;T=Float64)

Returns initial Q values for agent `θ`, converting to datatype `T`
"""
function init_Q(θ::Agent,T::DataType)
    return T.(θ.Q0)
end

"""
    init_Q(agents::Array{A};T=Float64) where {A <: Agent}

Wrapper for initializing Q values for multiple agents
"""
function init_Q(agents::Array{A}) where {A <: Agent}
    return reduce(vcat,init_Q.(agents[:])')
end

"""
    next_Q!(Q::AbstractArray{T},agents::Array{A},data::D,t) where {T <: Union{Float64,ForwardDiff.Dual}, A <: Agent, D <: RatData}

Wrapper for in-place updating all agent values after a single trial
"""
function next_Q!(Q::AbstractArray{T},agents::Array{A},data::D,t) where {T <: Union{Float64,ForwardDiff.Dual}, A <: Agent, D <: RatData}
    next_Q!.(eachrow(Q),agents[:],[data],[t])
end

"""
    Q_values(data::D,agents::Array{A}) where {A <: Agent, D <: RatData}

Computes Q values for all agents in `agents` using `data`. 
Returns 3D array of Q values, with dimensions (2,ntrials,nagents), where the first dimension corresponds to primary (1) and secondary (2) Q values.
"""
function Q_values(data::D,agents::Array{A}) where {A <: Agent, D <: RatData}
    @unpack choices,ntrials,sess_inds,new_sess = data
    na = length(agents)
    Q_all = Array{Float64}(undef,2,ntrials,na)
    for (a,agent) in enumerate(agents)
        for inds in sess_inds
            Q = Array{Float64}(init_Q(agent))
            for t in inds
                if new_sess[t]
                    Q .= init_Q(agent;data=data,t=t)
                end
                Q_all[:,t,a] = copy(Q[1:2])
                next_Q!(Q,agent,data,t)
            end
        end
    end
    return Q_all
end

"""
    simulate(options::AgentOptions) 

Simulates agents using parameter values specified by `options`. Returns new vector of agents with simulated parameter values.
"""
function simulate(options::AgentOptions) 
    @unpack agents,fit_symbs,symb_inds = options
    fit_priors = [Symbol(string(s)*"prior") for s in fit_symbs]
    params = zeros(size(fit_symbs))
    for (i,a) in zip(1:length(fit_symbs),symb_inds)
        params[i] = rand(getfield(agents[a],fit_priors[i]))
    end
    agents_sim = deepcopy(agents)
    update!(agents_sim,params,options)
    return agents_sim       
end

#####################################################
########--       helpful functions         --########
#####################################################
"""
    get_params(agents::Array{A},options::AgentOptions) where A <: Agent

Collects all parameters being fit from `agents` according to `options`. Unpacks `options` and calls `get_params(agents,symb_inds,fit_symbs)`
"""
function get_params(agents::Array{A},options::AgentOptions) where A <: Agent
    @unpack symb_inds,fit_symbs = options
    return get_params(agents,symb_inds,fit_symbs)
end

function get_params(agents::Array{A},options::AgentOptions,a::T) where {A <: Agent, T}
    # get params for agents specified by a
    params = get_params(agents,options)
    @unpack fit_params,param_inds = options
    if T <: Real
        i = fit_params[param_inds .== a]
    elseif T <: AbstractVector
        i_tmp = sum([param_inds .== ai for ai in a])
        i = fit_params[i_tmp .> 0]
    end
    return params[i]
end

"""
    get_params(agents::Array{A},symb_inds::Vector{Int64},fit_symbs::Vector{Symbol}) where A <: Agent

Pulls parameters from `agents` specified by `symb_inds` and `fit_symbs`
"""
function get_params(agents::Array{A},symb_inds::Vector{Int64},fit_symbs::Vector{Symbol}) where A <: Agent
    return getproperty.(agents[symb_inds],fit_symbs)
end

function get_params(agents::Array{A},symb_inds::Nothing,fit_symbs::Nothing) where A <: Agent
    # catch function for when no parameters are being fit
    return "nothing"
end

"""
    get_param(agent::A, symb::Symbol) where A <: Agent

Gets parameter `symb` from `agent`
"""
function get_param(agent::A,symb::Symbol) where A <: Agent
    return getproperty(agent,symb)
end

"""
    get_param(agents::Array{A}, symb::Symbol) where A <: Agent

Gets parameter `symb` from each agent in `agents`
"""
function get_param(agents::Array{A},symb::Symbol) where A <: Agent
    return getproperty.(agents,symb)
end


function get_priors(agents::Array{A},options::AgentOptions) where A <: Agent
    # gets parameter priors for `agents` specified by `options`
    @unpack symb_inds,fit_priors = options
    return get_params(agents,symb_inds,fit_priors)
end

function get_scales(agents::Array{A},options::AgentOptions) where A <: Agent
    # gets parameter scales for `agents` specified by `options`
    @unpack symb_inds,fit_scales = options
    return get_params(agents,symb_inds,fit_scales)
end

"""
    agent_color(agent::A) where A <: Agent

Grab agent color
"""
function agent_color(agent::A) where A <: Agent
    return agent.color
end

"""
    agent_color_lite(agent::A) where A <: Agent

Grab agent color_lite
"""
function agent_color_lite(agent::A) where A <: Agent
    return agent.color_lite
end

"""
Sort duplicate agents by fit parameter. UNFINISHED
"""
function sort_agents!(agents::Array{A},options::AgentOptions) where A <: Agent
    @unpack fit_params,fit_symbs,param_inds = options
    agents_str = agents2string(agents)
    dupes = findall([sum(str .== agents_str) for str in agents_str] .> 1)
    catch_i = 0
    params = get_params(agents,options,dupes)
    if !isempty(dupes)
        while !isempty(dupes) && (catch_i < length(dupes))
            agent = agents_str[dupes[1]]
            dupes_i = findall(agents_str[dupes] .== agent)
            params_i = dup_params[dupes_i]
            asort = reverse(sortperm(params_i))
            
            catch_i += 1
        end
    end

end