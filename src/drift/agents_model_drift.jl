"""
    optimize(data, options)

Optimizes MoA-β (MoA-psytrack) parameters using by alernating between finding the maximum-likelihood
β via gradient descent and fitting the model hyperparameters via gradient descent with updated β    
until convergence
"""
function optimize(data::RatData,model_options::ModelOptionsDrift,agent_options::AgentOptions;init_hypers=true,save_model=false,save_path=nothing,save_iter=false,fit_num=nothing,seed=nothing)
    @unpack maxiter,tol,nstarts = model_options
    @unpack fit_params = agent_options
    
    all_models = Array{ModelDrift}(undef,nstarts)
    all_agents = Array{Array{Agent}}(undef,nstarts)
    all_xs = Array{Array{Float64}}(undef,nstarts)
    all_lls = Array{Float64}(undef,nstarts)
    y = initialize_y(data)
    start_i = 1
    throw = 0

    if save_model
        if isnothing(save_path)
            save_path = pwd()
        elseif !isdir(save_path)
            mkdir(save_path)
        end
    end
    if !isnothing(seed)
        Random.seed!(seed)
    end
    while start_i <= nstarts && throw < 3
        if init_hypers
            agents = initialize(agent_options)
            model_options = initialize_hypers(model_options,agents)
        end
        model,_,x = initialize(model_options,agents,data)

        iter = 1
        dll = 1.
        neg_counter = 0
        lls = Array{Float64}(undef,2)
        while iter <= maxiter && (dll > tol || dll < 0) && neg_counter < 5
            println("\r start = ",start_i,"; iteration ", iter, " of ", maxiter)
            model_fit,agents_fit,x_fit,lls[2] = @time optimize(model,agents,y,x,data,model_options,agent_options)
            if iter > 1
                dll = lls[2]-lls[1]
            end
            println("ll = ",round(lls[2],digits=7),"; dll = ",round(dll,digits=7))

            iter += 1
            if dll < 0 
                neg_counter += 1
                model.β .= mean([model.β,model_fit.β])
                model.σ .= mean([model.σ,model_fit.σ])
                model.σSess .= mean([model.σSess,model_fit.σSess])
                α1 = get_params(agents,agent_options)
                α2 = get_params(agents_fit,agent_options)
                α = mean([α1,α2])
                update!(x,agents_fit,α,agent_options,data)
            else
                neg_counter = 0
                model = deepcopy(model_fit)
                agents = deepcopy(agents_fit)
                x = copy(x_fit)
                lls[1] = copy(lls[2])
            end
        end
        if isinf(lls[1])
            continue
            throw += 1
        else
            if save_model & save_iter
                fname = make_fname(model_options,agent_options,num=fit_num,iter=start_i)
                fpath = joinpath(save_path,fname)
                savevars(fpath;model,agents,model_options,agent_options,data,ll=lls[2])
            end

            all_lls[start_i] = copy(lls[1])
            all_models[start_i] = deepcopy(model)
            all_agents[start_i] = deepcopy(agents)
            all_xs[start_i] = copy(x)
            start_i += 1
        end
    end
    
    ll, best_fit = findmax(all_lls)
    model = all_models[best_fit]
    agents = all_agents[best_fit]
    x = all_xs[best_fit] 

    if save_model & !save_iter
        fname = make_fname(model_options,agent_options,num=fit_num)
        fpath = joinpath(save_path,fname)
        savevars(fpath;model,agents,model_options,agent_options,data,ll=lls[2])
    end
    
    return model,agents,y,x,ll
end

"""
    initialize(data, options)

Initializes model parameters, x, and y

"""

function initialize(options::ModelOptionsDrift,agents,data::RatData)
    @unpack choices,forced,nfree,new_sess_free = data
    nfeats = length(agents)
    y = choices .== 1
    deleteat!(y,findall(forced))
    x = initialize_x(agents,data)

    model = initialize(options,agents;ntrials=nfree,new_sess=new_sess_free)
    return model,y,x
end

"""
    initialize(options)

Initializes model parameters

"""

function initialize(options::ModelOptionsDrift,agents;ntrials=1,new_sess=nothing)
    @unpack σ0, σInit0, σSess0 = options
    if isnothing(new_sess)
        new_sess = falses(ntrials)
        new_sess[1] = true
    end
    
    β = Array{Float64}(undef,length(agents),ntrials)
    σ = copy(σ0)
    if length(σ) == 1
        σ = repeat(σ,length(agents))
    end
    σInit = copy(σInit0)
    if length(σInit) == 1
        σInit = repeat(σInit,length(agents))
    end
    σSess = copy(σSess0)
    if length(σSess) == 1
        σSess= repeat(σSess,length(agents))
    end

    bound = σInit .* 1.5

    β[:,1] .= rand.(Normal.(0,σInit))
    β[β[:,1] .> bound,1] = bound[β[:,1] .> bound]
    β[β[:,1] .< -bound,1] = -bound[β[:,1] .< -bound]

    force_pos = hasproperty.(agents,:α)
    β[force_pos,1] .= abs.(β[force_pos,1])


    for t = 2:ntrials
        if new_sess[t]
            β[:,t] .= β[:,t-1] .+ rand.(Normal.(0,σSess))
        else
            β[:,t] .= β[:,t-1] .+ rand.(Normal.(0,σ))
        end
        if any(β[:,t] .> bound)
            β[β[:,t] .> bound,t] = 2 .* bound[β[:,t] .> bound] .- β[β[:,t] .> bound,t]
        end
        if any(β[:,t] .< -bound)
            β[β[:,t] .< -bound,t] = -2 .* bound[β[:,t] .< -bound] .- β[β[:,t] .< -bound,t]
        end
    end
    return ModelDrift(β,σ,σInit,σSess)
end

"""
    initialize_hypers!(options)

Reinitializes model hyperparameters
"""
function initialize_hypers(options::ModelOptionsDrift,agents::AbstractArray{A}) where {A <: Agent}
    @unpack σ0, σInit0, σSess0 = options
    if length(σ0) < length(agents)
        σ0 = zeros(length(agents))
    end
    if length(σInit0) < length(agents)
        σInit0 = zeros(length(agents))
    end
    if length(σSess0) < length(agents)
        σSess0 = zeros(length(agents))
    end
    for a = 1:length(agents)
        σ0[a] = abs(rand(Normal(0,0.01))) 
        σInit0[a] = agents[a].βprior[1].σ
        σSess0[a] = abs(rand(Normal(0,0.03))) 
    end

    return ModelOptions(σ0,σInit0,σSess0,options)

end

function simulate(options::ModelOptionsDrift,agents::Array{S};ntrials=1,new_sess=nothing) where S <: Agent
    options_sim = initialize_hypers(options,agents)
    return initialize(options_sim,agents;ntrials=ntrials,new_sess=new_sess)
end


