
###-- converting stuff to dicts that replace special character field names --###

"""
    ratdata2dict(data::RatData)

Convert RatData to a dictionary and save RatData type
"""
function ratdata2dict(data::D) where D <: RatData
    data_dict = Dict(string(key)=>getfield(data,key) for key in fieldnames(D))
    delete!(data_dict,"sess_inds_free")
    delete!(data_dict,"sess_inds")
    data_dict["type"] = string(D)
    return data_dict
end

function model2dict(model::M,options::T) where {M <: MixtureAgentsModel, T <: ModelOptions}
    return merge(model2dict(model),Dict("ops"=>options2dict(options)))
end

function model2dict(model::ModelHMM)
    return Dict("beta"=>Array(model.β),"pi"=>Array(model.π),"A"=>Array(model.A))
end

function model2dict(model::ModelDrift)
    return Dict("beta"=>model.β,"sigma"=>model.σ,"sigmaInit"=>model.σInit,"sigmaSess"=>model.σSess)
end

function options2dict(options::T) where {T <: ModelOptions}
    @unpack nstarts,maxiter,tol = options
    options_dict = Dict{String,Any}()
    @pack! options_dict = nstarts,maxiter,tol
    options2dict!(options_dict,options)
    return options_dict
end

function options2dict!(options_dict,options::ModelOptionsHMM)
    @unpack β0,A0,π0,βprior,α_π,α_A,nstates = options
    options_dict["beta0"] = Array(β0)
    options_dict["A0"] = Array(A0)
    options_dict["pi0"] = Array(π0)
    options_dict["beta_prior"] = βprior
    options_dict["alpha_pi"] = Array(α_π)
    options_dict["alpha_A"] = Array(α_A)
    @pack! options_dict = nstates
end

function options2dict!(options_dict,options::ModelOptionsDrift)
    @unpack σ0,σInit0,σSess0 = options
    # return merge!
    options_dict["sigma0"] = σ0
    options_dict["sigmaInit0"] = σInit0
    options_dict["sigmaSess0"] = σSess0
end

function options2dict(options::AgentOptions)
    @unpack agents = options
    return agents2dict(agents,options)
end

function agents2dict(agents::AbstractArray{A},options::AgentOptions) where A <: Agent
    @unpack fit_symbs,fit_priors = options
    agents_dict = Dict("agents"=>agents2string(agents),"params"=>get_params(agents,options),"fit_params"=>get_fit_params(options),"fit_symbs"=>string.(fit_symbs),"beta_priors"=>prior2dict.(get_param(agents,:βprior)))
    if !isnothing(fit_priors)
        agents_dict["fit_priors"] = string.(fit_priors)
        agents_dict["param_priors"] = prior2dict(get_priors(agents,options))
    end
    return agents_dict
end

prior2dict(prior::Beta) = Dict("type"=>"Beta","params"=>[prior.α,prior.β])
prior2dict(prior::Normal) = Dict("type"=>"Normal","params"=>[prior.μ,prior.σ])
prior2dict(priors::Vector{D}) where {D <: Distribution} = prior2dict.(priors)

function options2dict(options::SimOptions)
    @unpack nsess,ntrials,mean_ntrials,std_ntrials,ntrials_for_flip,flip_prob,p_reward,p_congruent = options
    options_dict = Dict{String,Any}()
    @pack! options_dict = nsess,ntrials,mean_ntrials,std_ntrials,ntrials_for_flip,flip_prob,p_reward,p_congruent
    return options_dict
end


###-- converting stuff back to structs --###

function dict2model(model_dict)
    if haskey(model_dict,"pi")
        @unpack beta,A,pi = model_dict
        if typeof(beta) <: Dict
            beta = beta["data"]
            A = A["data"]
        end
        if length(size(beta)) < 2
            nstates = length(model_dict["pi"])
            beta = Matrix(reshape(beta,(:,nstates)))
            A = Matrix(reshape(A,(:,nstates)))
            pi = Array(pi)
        elseif size(model_dict["beta"],2) < 2
            A = [model_dict["A"];;]
            pi = [model_dict["pi"]]
        end
        model = ModelHMM(β=beta,π=pi,A=A)

    elseif haskey(model_dict,"sigma")
        @unpack beta,sigma,sigmaInit,sigmaSess = model_dict
        if length(size(beta)) < 2
            nagents = length(sigma)
            beta = Matrix(reshape(beta,(nagents,:)))
            sigma = Array(sigma)
            sigmaInit = Array(sigmaInit)
            sigmaSess = Array(sigmaSess)
        end
        model = ModelDrift(β=beta,σ=sigma,σInit=sigmaInit,σSess=sigmaSess)
    else
        error("Unrecognized model dictionary")
    end
    if haskey(model_dict,"ops")
        ops = dict2options(model_dict["ops"])
    else
        ops = ModelOptions(model)
    end
    return model,ops
end

function dict2options(ops_dict)
    @unpack nstarts,maxiter,tol = ops_dict
    if haskey(ops_dict,"nstates")
        @unpack nstates,beta0,A0,pi0 = ops_dict
        if typeof(beta0) <: Dict
            beta0 = beta0["data"]
        end
        if typeof(A0) <: Dict
            A0 = A0["data"]
        end
        if typeof(pi0) <: Dict
            pi0 = pi0["data"]
        end
        if length(size(beta0)) < 2
            if iszero(beta0)
                beta0 = zeros(1,nstates)
            else
                beta0 = Matrix(reshape(beta0,(:,nstates)))
            end
            if iszero(A0)
                A0 = zeros(nstates,nstates)
            else
                A0 = Matrix(reshape(A0,(:,nstates)))
            end
            if iszero(pi0)
                pi0 = zeros(nstates)
            else
                pi0 = Array(pi0)
            end
        end
        if nstates == 1
            A0 = [A0;;]
            pi0 = [pi0]
        end
        if haskey(ops_dict,"beta_prior")
            @unpack beta_prior,alpha_pi,alpha_A = ops_dict
            if nstates == 1
                alpha_pi = [alpha_pi]
                alpha_A = [alpha_A;;]
            end
            return ModelOptionsHMM(nstates=nstates,β0=beta0,A0=A0,π0=pi0,βprior=beta_prior,α_π=alpha_pi,α_A=alpha_A,nstarts=nstarts,maxiter=maxiter,tol=tol)
        else
            return ModelOptionsHMM(nstates=nstates,β0=beta0,A0=A0,π0=pi0,nstarts=nstarts,maxiter=maxiter,tol=tol)
        end
    elseif haskey(ops_dict,"sigma0")
        @unpack sigma0,sigmaInit0,sigmaSess0 = ops_dict
        if !(typeof(sigma0) <: Array)
            sigma0 = Array(sigma0)
        end
        if !(typeof(sigmaInit0) <: Array)
            sigmaInit0 = Array(sigmaInit0)
        end
        if !(typeof(sigmaSess0) <: Array)
            sigmaSess0 = Array(sigmaSess0)
        end
        return ModelOptionsDrift(σ0=sigma0,σInit0=sigmaInit0,σSess0=sigmaSess0,nstarts=nstarts,maxiter=maxiter,tol=tol)
    else
        error("unrecognized options dictionary")
    end
end

function dict2agents(agents_dict)
    agents = string2agents(agents_dict["agents"])
    if haskey(agents_dict,"fit_priors")
        options = AgentOptions(agents,Symbol.(agents_dict["fit_symbs"]),agents_dict["fit_params"],fit_priors=true)
        param_priors = dict2prior(agents_dict["param_priors"])
        update_priors!(agents,param_priors,options)
    else       
        options = AgentOptions(agents,Symbol.(agents_dict["fit_symbs"]),agents_dict["fit_params"])
    end
    if !(typeof(agents_dict["params"]) <: String)
        update!(agents,agents_dict["params"],options)
    end
    if haskey(agents_dict,"beta_priors")
        βpriors = dict2prior(agents_dict["beta_priors"])
        update_priors!(agents,:βprior,βpriors)
    end

    return agents,options
end

dict2prior(prior_dict::Dict) = prior_strings(prior_dict["type"])(prior_dict["params"]...)
dict2prior(prior_dict::Vector) = dict2prior.(prior_dict)

function dict2ratdata(data_dict)
    if typeof(collect(keys(data_dict))[1]) <: String
        data_dict = Dict(Symbol(key)=>value for (key,value) in data_dict)
    end
    type = split(data_dict[:type],"{")[1]
    delete!(data_dict,:type)
    return ratdata_tasks(type)(data_dict)
end

function fit2options(model::M,model_ops::O, agents::AbstractArray{A},agent_ops::AgentOptions) where {A <: Agent, M <: MixtureAgentsModel, O <: ModelOptions}
    model_options = ModelOptions(model,model_ops)
    agent_options = AgentOptions(agents,agent_ops)
    return model_options,agent_options
end


###-- convert agents to strings and vice versa --###
"""
    agents2string(agents::Vector{A}) where A <: Agent

Converts vector of Agents into vector of strings
"""
function agents2string(agents::Vector{A}) where A <: Agent
    return [agent2string(agent) for agent in agents]
end

"""
    string2agents(agents_str::AbstractArray{T}) where T <: Any

Converts vector of strings into vector of Agents
"""
function string2agents(agents_str::AbstractArray{T}) where T <: Any
    #agents_list = split(agents_str,"-")
    nagents = length(agents_str)
    agents = Array{Agent}(undef,nagents)
    for (a,str) in enumerate(agents_str)
        if any(contains.(str,[r"CR\d",r"UR\d",r"CO\d",r"UO\d"]))
            nback = parse(Int,str[3:end])
            agents[a] = agent_strings(str[1:2])(nback)
        elseif any(contains.(str,[r"R\d"]))
            nback = parse(Int,str[2:end])
            agents[a] = Reward(nback)
        elseif any(contains.(str,[r"C\d"]))
            nback = parse(Int,str[2:end])
            agents[a] = Choice(nback)
        else
            agents[a] = agent_strings(str)()
        end

    end
    return agents
end

"""
    string2agents(agents_str::T) where T <: String

Converts list of agents (concatenated string) into vector of Agents
"""
function string2agents(agents_list::T) where T <: AbstractString
    agents_str = string.(split(agents_list,"-"))
    agents = string2agents(agents_str)
    return agents
end


"""
    agents2list(agents::Vector{A}) where A <: Agent

Converts vector of Agents into list of agents (concatenated string)
"""
function agents2list(agents::Vector{T}) where T <: Agent
    agents_list = ""
    for agent in agents
        agents_list *= "-"*agent2string(agent)
    end
    return agents_list[2:end]
end
function agents2list(agents::Matrix{T}) where T <: Agent
    return agents2list(agents[:,1])
end

"""
    string2agents(agents_str)

Converts vector of agent strings into list of agents (concatenated string)
"""
function agents2list(agents::AbstractArray{T}) where T <: String
    agents_list = ""
    for agent in agents
        agents_list = string(agents_list,"-",agent)
    end
    return agents_list[2:end]
end