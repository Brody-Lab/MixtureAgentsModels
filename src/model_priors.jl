function prior_strings(prior::Union{AbstractString,Nothing}=nothing)
    # Dictionary of available prior types
    priors = Dict("Beta"=>Beta,"Normal"=>Normal,"Dirichlet"=>Dirichlet)
    if !isnothing(prior)
        return priors[prior]
    else
        return priors
    end
end

function logprior(prior::B,x) where B <: Beta
    # log-prior for Beta distribution
    return (prior.α - 1) * log(x) + (prior.β - 1) * log(1 - x)
end

function logprior(prior::N,x) where N <: Normal
    # log-prior for Normal distribution
    return -0.5 * (1 / prior.σ^2) * (x - prior.μ)^2
end

function logprior(prior::D,x) where D <: Dirichlet
    # log-prior for Dirichlet distribution
    return nansum(@. (prior.alpha - 1) * log(x))
end

function estimate_prior(prior::Type{Beta},x::T) where T <: AbstractVector
    # Uses method of moments to estimate shaping parameters for Beta prior. Adds 1 to each parameter if either is less than 1.
    α = moment_α(mean(x),var(x))
    β = α*(1-mean(x))/mean(x)
    if any([α,β] .< 1)
        α += 1
        β += 1
    end
    return prior(α,β)
end

function estimate_prior(prior::Type{Beta},x::A) where A <: AbstractArray
    # Estimates Beta prior for each row of `x`
    return estimate_prior.(prior,eachrow(x))
end

function estimate_prior(prior::Type{Normal},x) 
    # Estimates Normal prior using mean and standard deviation of `x`
    return prior(mean(x),std(x))
end

function estimate_prior(prior::Type{Dirichlet},x::A) where A <: AbstractArray
    # Uses method of moments to estimate shaping parameters for Dirichlet prior on elements of array `x`. Mean and standard deviation are computed along the final dimension and returns vector/array matching ndims(x)-1. Adds 1 to each parameter if any parameter is less than 1.
    α = abs.(dropdims(moment_α(mean(x,dims=ndims(x)),var(x,dims=ndims(x))),dims=ndims(x)))
    α[isnan.(α)] .= 0
    if any(α .< 1)
        α .+= 1
    end
    return α
end

function moment_α(μ,σ)
    # Element-wise method of moments for Beta and Dirichlet distributions
    return @. μ * (μ * (1 - μ)/σ - 1)
end

function population_priors(all_models,all_agents,model_options::ModelOptionsHMM,agent_options::AgentOptions)
    all_betas = cat(map((x)->x.β,all_models)...,dims=3)
    βmean = dropdims(mean(all_betas,dims=3),dims=3)
    βpriors = dropdims(mapslices((x)->estimate_prior(Normal,x),all_betas,dims=3),dims=3)

    all_pis = cat(map((x)->round.(x.π,digits=4),all_models)...,dims=2)
    πmean = dropdims(mean(all_pis,dims=2),dims=2)
    πpriors = estimate_prior(Dirichlet,all_pis)
    
    all_As = cat(map((x)->x.A,all_models)...,dims=3)
    Amean = dropdims(mean(all_As,dims=3),dims=3)
    Apriors = estimate_prior(Dirichlet,all_As)

    model_ops = ModelOptions(βmean,πmean,Amean,πpriors,Apriors,model_options)

    all_params = cat(map((a)->get_params(a,agent_options),all_agents)...,dims=2)
    param_means = dropdims(mean(all_params,dims=2),dims=2)
    param_priors = estimate_prior(Beta,all_params)

    agents_prior = Array{Agent}(undef,length(agent_options.agents))
    if isnothing(agent_options.fit_priors)
        agent_options = AgentOptions(agent_options.agents,agent_options.fit_symbs,get_fit_params(agent_options);fit_priors=true)
    end
    @unpack agents,fit_symbs,scale_x = agent_options
    for (a,agent) in enumerate(agents)
        agents_prior[a] = update(agent,:βprior,βpriors[a,:])
    end
    update_priors!(agents_prior,param_priors,agent_options)
    update!(agents_prior,param_means,agent_options)
    agent_ops = AgentOptions(agents_prior,fit_symbs,get_fit_params(agent_options);fit_priors=true,scale_x=scale_x)

    return model_ops,agent_ops
end

function update_priors!(agents::Array{A},priors::AbstractArray{T},options::AgentOptions) where {T <: Any, A <: Agent}
    @unpack fit_priors,fit_params,param_inds = options
    for (a,i) in zip(param_inds,fit_params)
        agents[a] = update(agents[a],fit_priors[i],priors[i])
    end
end

function update_priors!(agents::Array{A},symb::Symbol,priors::AbstractArray{T}) where {T <: Any, A <: Agent}
    for (a,p) in enumerate(priors)
        agents[a] = update(agents[a],symb,p)
    end
end