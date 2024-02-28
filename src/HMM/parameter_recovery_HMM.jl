"""
    parameter_recovery(sim_options::S, model_options::ModelOptionsHMM, agent_options::AgentOptions; nsims::Int=100, save_rec::Bool=false, save_path=nothing, fname=nothing) where S <: SimOptions

Simulates two-step data generated from model specified by model_options and fits model to simulated data. Repeats nsims times. Returns recovered parameters and log-likelihoods.

# Optional arguments:
- `nsims`: number of simulations (default = 100)
- `save_rec`: if `true`, save recovered parameters and log-likelihoods to file
- `save_path`: path to save file (default = pwd())
- `fname`: filename (default = "model_fit.jld2")
"""
function parameter_recovery(sim_options::S, model_options::ModelOptionsHMM, agent_options::AgentOptions; nsims::Int=100, save_rec::Bool=false, save_path=nothing, fname=nothing, overwrite=false, agents_prior=nothing, model_sim=nothing,agents_sim=nothing) where S <: SimOptions
    @unpack nstates = model_options
    @unpack agents,fit_symbs = agent_options
    
    β_recovery_all = Array{Float64}(undef,length(agents),nstates,2,nsims)
    π_recovery_all = Array{Float64}(undef,nstates,2,nsims)
    A_recovery_all = Array{Float64}(undef,nstates,nstates,2,nsims)
    α_recovery_all = Array{Float64}(undef,length(fit_symbs),2,nsims)    
    ll_recovery_all = Array{Float64}(undef,2,nsims)
    if save_rec
        if isnothing(save_path)
            save_path = pwd()
        elseif !isdir(save_path)
            mkdir(save_path)
        end
        if isnothing(fname)
            fname = split(make_fname(model_options,agent_options),".")
            fname = fname[1]*"_recovery."*fname[2]
        end
        fname = split(fname,".")
    end

    for sim in 1:nsims  
        println("sim "*string(sim))

        if save_rec 
            fname_sim = fname[1]*"_sim"*string(sim)*"."*fname[2]
            fpath = joinpath(save_path,fname_sim)
            if isfile(fpath) & !overwrite
                println("sim already saved, skipping")
                continue
            end
        end

        β_recovery,π_recovery,A_recovery,α_recovery,ll_recovery = simulate_and_fit(sim_options,model_options,agent_options;agents_prior=agents_prior,model_sim=model_sim,agents_sim=agents_sim)

        if save_rec
            savevars(fpath;beta_recovery=β_recovery,pi_recovery=π_recovery,A_recovery=A_recovery,alpha_recovery=α_recovery,ll_recovery=ll_recovery)
        end

        β_recovery_all[:,:,:,sim] = β_recovery
        π_recovery_all[:,:,sim] = π_recovery
        A_recovery_all[:,:,:,sim] = A_recovery
        α_recovery_all[:,:,sim] = α_recovery
        ll_recovery_all[:,sim] = ll_recovery

        GC.gc()
    end


    return β_recovery_all,π_recovery_all,A_recovery_all,α_recovery_all,ll_recovery_all
end

function simulate_and_fit(sim_options::S,model_options::ModelOptionsHMM,agent_options::AgentOptions;agents_prior=nothing,model_sim=nothing,agents_sim=nothing) where S <: SimOptions
    @unpack nstates = model_options
    @unpack agents,fit_symbs = agent_options

    ll_recovery = [0.,0.]

    if isnothing(model_sim) || isnothing(agents_sim)
        data,model_sim,agents_sim = simulate_task(sim_options,model_options,agent_options;agents_prior=agents_prior)
    else
        data = simulate_task(sim_options,model_sim,agents_sim)
    end

    model_fit,agents_fit,ll_fit = optimize(data,model_options,agent_options;disp_iter=10)

    if nstates > 1
        match_states!(model_fit,agents_fit,model_sim,agents_sim,data)
        # if size(agents,2) > 1
        #     agents_fit = agents_fit[:,zord]
        # end
    end

    if !isnothing(agent_options.fit_params)
        param_recovery = [get_params(agents_sim,agent_options) get_params(agents_fit,agent_options)]
    else
        param_recovery = nothing
    end
    β_recovery = cat(model_sim.β,model_fit.β,dims=3)
    π_recovery = [model_sim.π model_fit.π]
    A_recovery = cat(model_sim.A,model_fit.A,dims=3)

    _,_,ll_sim = compute_posteriors(model_sim,agents_sim,data)
    ll_recovery = [ll_sim,ll_fit]

    ll_recovery = exp.(ll_recovery ./ data.ntrials)

    return β_recovery,π_recovery,A_recovery,param_recovery,ll_recovery
end


"""
    match_latent_states!(model_fit::ModelHMM,model_sim::ModelHMM,nstates)

Finds the most-likely state alignment to match the latent states of `model_fit` to `model_sim`. Modifies `model_fit` in-place. Returns the new ordering of latent states.
"""
function match_latent_states!(model_fit::ModelHMM,model_sim::ModelHMM)
    nstates = length(model_sim.π)
    if length(model_fit.π) != nstates
        error("model_fit and model_sim must have same number of latent states")
    end

    if size(model_sim.β) != size(model_fit.β)
        sim_vectors = Array(vcat(model_sim.π',sort(model_sim.A,dims=2)'))
        fit_vectors = Array(vcat(model_fit.π',sort(model_fit.A,dims=2)'))  
    else
        sim_vectors = Array(vcat(model_sim.β,model_sim.π',sort(model_sim.A,dims=2)'))
        fit_vectors = Array(vcat(model_fit.β,model_fit.π',sort(model_fit.A,dims=2)'))
    end

    r = pairwise(CosineDist(), sim_vectors, fit_vectors, dims=2)
    zpairs = argmin(r,dims=2)
    zsort = vec(getindex.(zpairs,2))
    zrem = setdiff(1:nstates,zsort)

    iter = 0
    while !isempty(zrem) & (iter <= nstates)
        for z = 1:nstates
            if sum(zsort .== z) > 1
                zdup = findall(zsort .== z)
                zkeep = zdup[argmin(r[zpairs[zdup]])]
                zdup = setdiff(zdup,zkeep)
                znew = vec(getindex.(argmin(r[zdup,zrem],dims=2),2))
                zsort[zdup] .= zrem[znew]
            end
        end
        zrem = setdiff(1:nstates,zsort)
        iter += 1
    end    
    if !isempty(zrem)
        error("wut")
    end

    model_fit.β .= model_fit.β[:,zsort]
    model_fit.π .= model_fit.π[zsort]
    model_fit.A .= model_fit.A[zsort,zsort]

    return zsort
end

