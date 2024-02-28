"""
    model_comparison(data,models,nfold,seed)

Run nfold cross-validation on data for mutlitple models specified in `models`
"""
function model_compare(data::RatData,model_ops::Array{M},agent_ops::Array{A};nfold=6,sim=nothing,cv_kwargs...) where {M <: ModelOptions, A <: AgentOptions}

    nmodels = length(model_ops)
    ll_train = Array{Float64}(undef,nmodels)
    ll_test = Array{Float64}(undef,nmodels)
    ll_train_n = Array{Float64}(undef,nmodels,nfold)
    ll_test_n = Array{Float64}(undef,nmodels,nfold)

    for ((m,model_op),agent_op) in collect(zip(enumerate(model_ops),agent_ops))
        fname = make_fname(model_op,agent_op,nfold=nfold,sim=sim)

        _,_,ll_train[m],ll_test[m],ll_train_n[m,:],ll_test_n[m,:] = cross_validate(data,model_op,agent_op;nfold=nfold,fname=fname,cv_kwargs...)

    end

    return ll_train,ll_test,ll_train_n,ll_test_n

end

"""
    nstates_comparison(options,nstates_range)

Compute a set of models that differ by the number of latent states.
Other model options specified in `options`
"""
function nstates_comparison(options::ModelOptionsHMM,nstates_range::UnitRange{Int64}=1:6) 
    nmodels = length(nstates_range)
    model_ops = Array{ModelOptionsHMM}(undef,nmodels)
    @unpack nstarts,maxiter,tol,βprior = options

    for (m,nstates) in zip(1:nmodels,nstates_range)
        model_ops[m] = ModelOptionsHMM(nstates=nstates,nstarts=nstarts,maxiter=maxiter,tol=tol,βprior=βprior)
    end
    
    return model_ops
end

"""
    nstates_comparison(options)

Compute a set of models that leaves out one of each agent specified in `options`
First model in set is all agents (original model) specified in `options`
"""
function agents_comparison(model_options::T,agent_options::AgentOptions) where T <: ModelOptions
    @unpack agents,fit_params,fit_symbs,fit_scales,fit_priors,scale_x = agent_options
    nagents = length(agents)
    nmodels = nagents + 1
    model_ops = Array{T}(undef,nmodels)
    model_ops[1] = deepcopy(model_options)
    agent_ops = Array{AgentOptions}(undef,nmodels)
    agent_ops[1] = deepcopy(agent_options)

    for (m,a) in zip(2:nmodels,1:nagents)
        agents_tmp = deepcopy(agents)
        deleteat!(agents_tmp,a)
        fit_params_tmp = get_fit_params(agent_options)
        deleteat!(fit_params_tmp,a)
        param_inds_tmp = get_param_inds(fit_params_tmp)
        fit_params_tmp = reduce(vcat,fit_params_tmp)
        fit_symbs_tmp = copy(fit_symbs[unique(fit_params_tmp[fit_params_tmp.!=0])])
        fit_scales_tmp = copy(fit_scales[unique(fit_params_tmp[fit_params_tmp.!=0])])
        if !isnothing(fit_priors)
            fit_priors_tmp = copy(fit_priors[unique(fit_params_tmp[fit_params_tmp.!=0])])
        else
            fit_priors_tmp = nothing
        end
        if isempty(fit_symbs_tmp)
            fit_symbs_tmp = nothing
            fit_params_tmp = nothing
            fit_scales_tmp = nothing
            fit_priors_tmp = nothing
        else
            fit_params_tmp[fit_params_tmp .> a] .-= 1
        end
        agent_ops[m] = AgentOptions(agents_tmp,fit_symbs_tmp,fit_params_tmp,param_inds_tmp,fit_scales_tmp,fit_priors_tmp,scale_x)
        if size(model_options.β0,1) > 1
            model_ops[m] = remove_agent(model_options,a)
        else
            model_ops[m] = deepcopy(model_options)
        end
    end

    return model_ops,agent_ops
end

function remove_agent(options::ModelOptionsHMM,a::Int)
    @unpack β0 = options
    if size(β0,1) != 1
        β0_tmp = copy(β0[1:end .!= a,:])
    else
        β0_tmp = copy(β0)
    end
    return Parameters.reconstruct(options,β0=β0_tmp)
end

function remove_agent(options::ModelOptionsDrift,a::Int)
    @unpack σ0,σInit0,σSess0 = options
    if length(σ0) != 1
        σ0_tmp = copy(σ0[1:end .!= a])
    else
        σ0_tmp = copy(σ0)
    end
    if length(σInit0) != 1
        σInit0_tmp = copy(σInit0[1:end .!= a])
    else
        σInit0_tmp = copy(σInit0)
    end
    if length(σSess0) != 1
        σSess0_tmp = copy(σSess0[1:end .!= a])
    else
        σSess0_tmp = copy(σSess0_tmp)
    end
    return Parameters.reconstruct(options,σ0=σ0_tmp,σInit0=σInit0_tmp,σSess0=σSess0_tmp)
end



