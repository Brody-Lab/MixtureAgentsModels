"""
    optimize(data::D,model_options::ModelOptionsHMM,agent_options::AgentOptions;init_hypers::Bool=true,seed=nothing,save_model=false,save_path=nothing,save_iter=false,fit_num=nothing)

Optimizes MoA-HMM parameters using expectation-maximization

# Required Inputs:
- `data::D<:RatData`: RatData struct contating behavioral data
- `model_options::ModelOptionsHMM`: HMM model options struct
- `agent_options::AgentOptions`: agent options struct

# Optional arguments:
- `data_test<:Union{RatData,nothing}=nothing`: optional held-out RatData struct for model validation
- `verbose::Bool=true`: whether to print progress
- `disp_iter::Int=1`: how often to print progress
- `init_hypers::Bool=true`: whether to initialize hyperparameters
- `init_beta::Bool=false`: whether to initialize β parameters
- `seed::Int=nothing`: random seed for initialization
- `save_model::Bool=false`: whether to save model
- `save_path::String=nothing`: path to save model
- `fname::String=nothing`: file name to save model
- `overwrite::Bool=false`: whether to overwrite existing file
- `save_iter::Bool=false`: whether to save model at each iteration
- `fit_num::Int=nothing`: number of fit to append to save file name

# Examples:
See `examples/example_fit_HMM.jl` for example implementation
"""
function optimize(data::RatData,model_options::ModelOptionsHMM,agent_options::AgentOptions;data_test::Union{RatData,Nothing}=nothing,verbose::Bool=true,disp_iter::Int=1,init_hypers::Bool=true,init_beta::Bool=true,seed::Int=0,save_model::Bool=false,save_path=nothing,fname=nothing,overwrite=false,save_iter::Bool=false,fit_num=nothing)::Tuple{ModelHMM,Array{Agent},Vararg{Float64}} 
    @unpack maxiter,tol,nstarts = model_options
    @unpack fit_params = agent_options

    # if saving, create directory, check if file already exists
    if save_model
        if isnothing(save_path)
            save_path = pwd()
        elseif !isdir(save_path)
            mkpath(save_path)
        end
        if isnothing(fname)
            fname = make_fname(model_options,agent_options,num=fit_num)
        end
        fpath = joinpath(save_path,fname)
        if isfile(fpath) & !overwrite            
            model_load,_,agents_load,_,data_load,ll_load,_,ll_test_load = loadfit(fpath)
            # make sure fit was saved with same data
            if all(data_load.choices .== data.choices)
                println(fname," already created, loading instead")
                if !isnothing(data_test)
                    return model_load, agents_load, ll_load, ll_test_load
                else
                    return model_load, agents_load, ll_load
                end
            else
                println(fname," already exists for different data, setting overwrite to true")
                overwrite = true
            end
        end
    end

    # initialize parameters for all starts
    if seed > 0
        Random.seed!(seed)
    end
    all_lls = zeros(nstarts)
    # all_lls_test = zeros(nstarts)
    all_agents = Array{Array{Agent}}(undef,nstarts)
    all_models = Array{ModelHMM}(undef,nstarts)

    y = initialize_y(data)
    #Threads.@threads 
    Threads.@threads for start_i = 1:nstarts

        # if saving each iteration, check if file exists. if it does exist, load it instead of fitting
        if save_model & save_iter
            fname_i = fname[1:end-4]*"_i"*string(start_i)*fname[end-3:end]
            fpath_i = joinpath(save_path,fname_i)
            if isfile(fpath_i) & !overwrite
                model_i,_,agents_i,_,data_i,ll_i = loadfit(fpath_i)
                # make sure iteration was saved with same data
                if all(data_i.choices .== data.choices)
                    println(fname_i," already created, loading instead")
                    all_models[start_i] = model_i
                    all_agents[start_i] = agents_i
                    all_lls[start_i] = ll_i
                    continue
                end
            end
        end

        if init_hypers
            agents_i = initialize(agent_options)
            model_i = initialize(initialize_hypers(model_options,agents_i;init_beta=init_beta),agents_i)
        else
            agents_i = deepcopy(agent_options.agents)
            model_i = initialize(model_options,agents_i)
        end

        x = initialize_x(data,agents_i)

        iter = 1
        dll = 1.
        dll_test = 1.
        wdiff = 1.
        pdiff = 1.
        πdiff = 1.
        Adiff = 1.
        neg_counter = 0
        lls = Array{Float64}(undef,2)
        lls_test = Array{Float64}(undef,2)
        while iter <= maxiter && (dll > tol || dll < 0) && (wdiff > tol || pdiff > tol) && !isnan(dll) && neg_counter < 11 #(wdiff > tol || pdiff > tol) &&
            if verbose && ((iter % disp_iter == 0) || (iter == 1))
                str = string("start = ",start_i,"; iteration ", iter, " of ", maxiter)
                println(str)
                model_fit, agents_fit, lls[2], lls_test[2] = @time optimize(model_i,agents_i,y,x,data,model_options,agent_options,data_test)
            else
                model_fit, agents_fit, lls[2], lls_test[2] = optimize(model_i,agents_i,y,x,data,model_options,agent_options,data_test)
            end
            
            if iter > 1
                dll = lls[2]-lls[1]
                dll_test = lls_test[2]-lls_test[1]
            end
            wdiff = maximum(abs.(model_fit.β - model_i.β))
            πdiff = maximum(abs.(model_fit.π - model_i.π))
            Adiff = maximum(abs.(model_fit.A - model_i.A))
            if !isnothing(fit_params)
                pdiff = maximum(abs.(get_params(agents_fit,agent_options) - get_params(agents_i,agent_options)))
            else
                pdiff = 0.
            end
            if verbose && ((iter % disp_iter == 0) || (iter == 1))
                str = string("  wdiff = ",round(wdiff,digits=5),"; pdiff = ",round(pdiff,digits=5),
                    "\n  πdiff = ",round(πdiff,digits=5),"; Adiff = ",round(Adiff,digits=5),
                    "\n  ll = ",round(lls[2],digits=5),"; dll = ",round(dll,digits=5),
                    # "\n  ll_test = ",round(lls_test[2],digits=5),"; dll_test = ",round(dll_test,digits=5)
                    )
                println(str)
            end
            iter += 1
            # check for overfitting if using a test set, or decreasing quality of train fit
            if (dll_test < 0) || (dll < 0)
                neg_counter += 1
            else
                neg_counter = 0
            end 
            model_i = deepcopy(model_fit)
            agents_i = deepcopy(agents_fit)
            lls[1] = copy(lls[2])
            lls_test[1] = copy(lls_test[2])
        end
        if save_model & save_iter
            # if !isnothing(data_test)
            #     savefit(fpath_i,model_i,model_options,agents_i,agent_options,data,lls[2],data_test,lls_test[2])
            # else
            savefit(fpath_i,model_i,model_options,agents_i,agent_options,data,lls[2])
            # end
        end
        all_lls[start_i] = copy(lls[2])
        # all_lls_test[start_i] = copy(lls_test[2])
        all_models[start_i] = model_i
        all_agents[start_i] = agents_i

    end
    bads = isnan.(all_lls)
    deleteat!(all_lls,bads)
    deleteat!(all_models,bads)
    deleteat!(all_agents,bads)
    ll, best_fit = findmax(all_lls)
    model = all_models[best_fit]
    agents = all_agents[best_fit]

    if save_model
        # if !isnothing(data_test)
        #     savefit(fpath,model,model_options,agents,agent_options,data,ll,data_test,ll_test)
        # else
        savefit(fpath,model,model_options,agents,agent_options,data,ll)
        # end
    end

    # if !isnothing(data_test)
    #     return model, agents, ll, ll_test
    # else
    return model, agents, ll
    # end
end

"""
    mean_model(models::Vector{ModelHMM})

Returns a new model with fields set to the mean parameters of a vector of models 
"""
function mean_model(models::Vector{ModelHMM}) 
    @unpack β,A,π = models[1]
    β = zeros(size(β))
    A = zeros(size(A))
    π = zeros(size(π))
    for m in models
        β .+= m.β
        A .+= m.A
        π .+= m.π
    end
    β ./= length(models)
    A ./= length(models)
    π ./= length(models)
    return ModelHMM(β,π,A)
end

"""
    initialize(data::D, options::ModelOptionsHMM, agents::Array{A}) where {D <: RatData, A <: Agent}

Initializes `model`, `x`, and `y` given `data`, `agents`, and `options` 
"""
function initialize(options::ModelOptionsHMM,agents::Array{A},data::D) where {D <: RatData, A <: Agent}
    y,x = initialize(data,agents)
    model = initialize(options,agents)
    return model, y, x
end

"""
    initialize(options,agents)

Initializes model parameters given `options`. Optional fields are only for compatibility with functions that also call version for ModelDrift
"""
function initialize(options::ModelOptionsHMM,agents;kwargs...)
    @unpack nstates,β0,A0,π0 = options

    β = copy(β0)
    if size(β,1) == 1
        β = SizedMatrix{length(agents),nstates}(repeat(β,size(agents,1)))
    end
    if !iszero(A0)
        A = copy(A0)
    else
        _,A,_ = initialize_hypers(options,agents;return_vars=true)
    end
    if !iszero(π0)
        π = copy(π0)
    else
        _,_,π = initialize_hypers(options,agents;return_vars=true)
    end

    return ModelHMM(β,π,A)
end

"""
    initialize_hypers(options,agents)

Reinitializes model hyperparameters given `options` and `agents`
"""
function initialize_hypers(options::ModelOptionsHMM,agents::Array{A};return_vars::Bool=false,init_beta::Bool=true) where {A <: Agent}
    @unpack nstates = options
    if init_beta
        β0 = SizedMatrix{length(agents),nstates}(rand(Normal(0,0.1),length(agents),nstates))
    else
        β0 = @SArray zeros(size(agents,1),nstates)
    end
    A0 = zeros(nstates,nstates)
    if nstates > 1
        # rs = rand(Beta(10,2),nstates)
        for i = 1:nstates
            αs = ones(nstates)
            αs[i] = 10
            A0[i,:] = rand(Dirichlet(αs))
        end
    else
        A0 = ones(1,1)
    end
    π0 = rand(Dirichlet(ones(nstates)))
    # π0 ./= sum(π0)

    if !return_vars
        return ModelOptions(β0,SizedVector{nstates}(π0),SizedMatrix{nstates,nstates}(A0),options)
    else
        return β0,A0,π0
    end
end

function simulate(options::ModelOptionsHMM,agents::Array{S};kwargs...) where S <: Agent
    @unpack nstates,α_A,α_π = options
    nagents = length(agents)
    β = zeros(nagents,nstates)
    for (a,agent) in enumerate(agents)
        if length(agent.βprior) == 1
            β[a,:] = rand(agent.βprior[1],nstates)
        else
            β[a,:] = rand.(agent.βprior)
        end
    end
    A = zeros(nstates,nstates)
        
    if nstates > 1
        if any(α_A .> 1)
            A .= permutedims(reduce(hcat,rand.(Dirichlet.(eachrow(α_A)))))
        else
            # rs = rand(Beta(10,2),nstates)
            for i = 1:nstates
                αs = ones(nstates)
                αs[i] = 10
                A[i,:] = rand(Dirichlet(αs))
            end
        end

        π = rand(Dirichlet(α_π))

    else
        A = ones(1,1)
        π = [1]
    end

    # π = rand(Dirichlet(ones(nstates)))

    return ModelHMM(β,π,A)
end


"""
    match_states!(model_fit::ModelHMM,model_source::ModelHMM,nstates)

Finds the most-likely state alignment to match the latent states of `model_sort` to `model_sim`. Modifies `model_sort` in-place. Returns the new ordering of latent states.
"""
function match_states!(model_sort::ModelHMM,agents_sort,model_source::ModelHMM,agents_source,data)
    
    nstates = length(model_source.π)
    if length(model_source.π) != nstates
        error("model_sort and model_source must have same number of latent states")
    end

    # source_vectors = expected_state_likelihood(model_source,agents_source,data)
    source_vectors = compute_posteriors(model_source,agents_source,data)[1]
    sort_vectors = compute_posteriors(model_sort,agents_sort,data)[1]
    # sort_vectors = expected_state_likelihood(model_sort,agents_sort,data)

    zsort = match_vectors(sort_vectors',source_vectors')

    # zord = zeros(Int,nstates)
    # zord[zsort] .= 1:nstates

    # if !all(zord .== zsort)
    #     error("wut")
    # end

    # @unpack β,π,A = model_sort
    model_sort.β .= model_sort.β[:,zsort] # model_orig.β[:,getindex.(z_pairs,1)]
    model_sort.π .= model_sort.π[zsort]
    model_sort.A .= model_sort.A[zsort,zsort]
    if size(agents_sort,2) > 1
        agents_sort .= agents_sort[:,zsort]
    end

    return zsort
end

function match_vectors(sort_vectors,source_vectors)
    nstates = size(source_vectors,2)
    # r = pairwise(CosineDist(), source_vectors, sort_vectors, dims=2)
    r = cor(source_vectors,sort_vectors,dims=1)
    zpairs = argmax(r,dims=2)
    zsort = vec(getindex.(zpairs,2))
    zrem = setdiff(1:nstates,zsort)

    iter = 0
    while !isempty(zrem) & (iter <= nstates)
        for z = 1:nstates
            if sum(zsort .== z) > 1
                zdup = findall(zsort .== z)
                zkeep = zdup[argmax(r[zpairs[zdup]])]
                zdup = setdiff(zdup,zkeep)
                znew = vec(getindex.(argmax(r[zdup,zrem],dims=2),2))
                zsort[zdup] .= zrem[znew]
            end
        end
        zrem = setdiff(1:nstates,zsort)
        iter += 1
    end    
    if !isempty(zrem)
        error("wut")
    end

    return zsort
end
