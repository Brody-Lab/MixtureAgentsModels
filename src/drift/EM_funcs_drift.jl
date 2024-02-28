"""
    optimize(model, agents, y, x, data, model_options, agent_options)

A single pass on first finding maximum-likelihood β via gradient descent, then fitting the model
hyperparameters via gradient descent with updated β

"""


function optimize(model::ModelDrift,agents::Vector{T},y,x,data::RatData,model_options::ModelOptionsDrift,agent_options::AgentOptions) where T<:Agent
    # first optimize β
    @unpack new_sess_free = data
    @unpack β,σ,σInit,σSess = model
    @unpack fit_params,fit_priors = agent_options

    # optimize β
    βfit = optimize(β,model,y,x,new_sess_free)

    # optimize σ and α
    # vectorize parameters of interest
    if !isnothing(fit_params)
        params = get_params(agents,agent_options)
        scale = [2,2,get_scales(agents,agent_options)]
        vars = [σ,σSess,params]
        v0 = vectorize(vars,scale)
    else
        scale = [2,2]
        vars = [σ,σSess]
        v0 = vectorize(vars,scale)
    end

    model_tmp = ModelDrift(βfit,σ,σInit,σSess)
    x_fit = copy(x)
    agents_fit = deepcopy(agents)
    ℓℓ(v) = negloglikehyper(v,vars,scale,y,x_fit,model_tmp,agents_fit,agent_options,data)

    #res = Optim.optimize(ℓℓ, v0 .- 10, v0 .+ 10, v0, SAMIN(rt=0.5), Optim.Options(iterations=length(v0)))
    res = Optim.optimize(ℓℓ, v0, NelderMead(), Optim.Options(iterations=1))# length(v0)))

    v = Optim.minimizer(res)
    nll = Optim.minimum(res)

    # extract parameters of interest from output vector
    if !isnothing(fit_params)
        σfit,σSfit,params = extract(v,vars,scale)
        update!(x_fit,agents_fit,params,agent_options,data)
    else 
        σfit,σSfit = extract(v,vars,scale)
    end

    model_fit = ModelDrift(βfit,σfit,σInit,σSfit)

    return model_fit,agents_fit,x_fit,-nll
end

"""
    negloglikehyper(v, vars, sizes, lens, y, x, γs, agents, options, data)

Computes the negative log-likelihood of the model hyperparameters for gradient descent.

"""

function negloglikehyper(v::Vector{T},vars,scale,y,x,model::ModelDrift,agents::Vector{A},options::AgentOptions,data::RatData) where {T <: Real, A <: Agent}
    @unpack β,σInit=model
    @unpack fit_params = options
    @unpack new_sess_free = data
    xtmp = convert(Array{T},x)
    agents_tmp = deepcopy(agents)

    # extract parameters from vector input
    if !isnothing(fit_params)
        σfit,σSfit,params = extract(v,vars,scale)
        update!(xtmp,agents_tmp,params,options,data)
    else
        σfit,σSfit = extract(v,vars,scale)
    end

    model_tmp = ModelDrift(β,σfit,σInit,σSfit)
    ll = compute_evidence(model_tmp,y,xtmp,new_sess_free)

    return -ll
end

"""
    compute_evidence(model, y, x, new_sess)

Computes the total model evidence

"""

function compute_evidence(model::ModelDrift,y,x,new_sess)
    nv = length(x)
    logβ = zeros(typeof(model.σ[1]),1)
    logpy = zeros(typeof(model.σ[1]),1)
    ddlogβ = spzeros(typeof(model.σ[1]),nv,nv)
    ddlogpy = spzeros(typeof(model.σ[1]),nv,nv)
    compute_posteriors!(logβ,logpy,nothing,nothing,ddlogβ,ddlogpy,model,y,x,new_sess)
    log_post = (1/2) * logabsdet(ddlogβ + ddlogpy)[1]
    
    return logβ[1] + logpy[1] - log_post
end

"""
    compute_evidence(model, y, x, new_sess)

Computes the total model evidence

"""
function compute_evidence(model::ModelDrift,agents::Array{A},data::D) where {D <: RatData, A <: Agent}
    @unpack new_sess_free = data
    y,x = initialize(agents,data)
    return compute_evidence(model,y,x,new_sess_free)
end

"""
    optimize(β, model, y, x, new_sess)

Estimate maximum-likelihood β via gradient descent

"""

function optimize(β::AbstractArray,model::ModelDrift,y::AbstractVector{Bool},x::Array,new_sess::AbstractVector{Bool})
    v0 = vec(β)

    ℓℓ!(F,G,H,v) = negloglikeβ!(F,G,H,v,model,y,x,new_sess)
    res = Optim.optimize(Optim.only_fgh!(ℓℓ!), v0, LBFGS(linesearch = LineSearches.BackTracking()), Optim.Options(iterations=10000))

    # ℓℓ!(F,G,v) = negloglikeβ!(F,G,nothing,v,model,y,x,new_sess)
    # res = Optim.optimize(Optim.only_fg!(ℓℓ!), v0, ConjugateGradient(), Optim.Options(iterations=1000))



    v = Optim.minimizer(res)
    βfit = reshape(v,size(β))

    return βfit

end

"""
    negloglikeβ!(F, G, H, v, model, y, x, new_sess)

Computes negative log-likelihood of model while fitting β

"""

function negloglikeβ!(F,G,H,v::Vector{T},model::ModelDrift,y,x,new_sess) where T
    @unpack β,σ,σInit,σSess = model
    βfit = reshape(v,size(β))
    model_tmp = ModelDrift(βfit,σ,σInit,σSess)
    # model_tmp = ModelDrift(βfit,βInit,σ,σInit)

    nll = negloglikelihood!(F,G,H,model_tmp,y,x,new_sess)
    return nll
end

"""
    negloglikelihood!(F, G, H, model, y, x, new_sess)

In-place negative log-likelihood of model with container for gradient and hessian

"""

function negloglikelihood!(F,G,H,model::ModelDrift,y,x,new_sess)

    if G !== nothing
        dlogβ,dlogpy = compute_gradients(model,y,x,new_sess) 
        G .= -(dlogβ .+ dlogpy)
    end
    
    if H !== nothing
        ddlogβ,ddlogpy = compute_hessians(model,y,x,new_sess)
        H .= -(ddlogβ .+ ddlogpy)
    end

    if F !== nothing
        logβ,logpy = compute_posteriors(model,y,x,new_sess)
        return -(logβ .+ logpy)
    end

end

"""
    compute_posteriors(model, y, x, new_sess)

Compute posterior components of model

"""

function compute_posteriors(model::ModelDrift,y::AbstractVector{S},x,new_sess) where  {S <: Bool}
    # types = [T1,T2]
    # if any(types .!== Float64)
    #     # iT = findfirst(types .!== Float64)
    #     logβ = zeros(types[iT],1)
    #     logpy = zeros(types[iT],1)
    # else
    logβ = zeros(1)
    logpy = zeros(1)
    # end

    compute_posteriors!(logβ,logpy,nothing,nothing,nothing,nothing,model,y,x,new_sess)
    return logβ[1],logpy[1]
end

"""
    compute_gradient(model, y, x, new_sess)

Compute gradient components of model

"""

function compute_gradients(model::ModelDrift,y::AbstractVector{S},x,new_sess) where {S <: Bool}
    nv = length(x)
    dlogβ = zeros(nv)
    dlogpy = zeros(nv)
    compute_posteriors!(nothing,nothing,dlogβ,dlogpy,nothing,nothing,model,y,x,new_sess)
    return dlogβ,dlogpy
end

"""
    compute_hessian(model, y, x, new_sess)

Compute hessian components of model

"""

function compute_hessians(model::ModelDrift{T1,T2},y::AbstractVector{S},x,new_sess) where {T1 <: Any, T2 <: Any, S <: Bool}
    nv = length(x)
    ddlogβ = spzeros(T1,nv,nv)
    ddlogpy = spzeros(T1,nv,nv)
    compute_posteriors!(nothing,nothing,nothing,nothing,ddlogβ,ddlogpy,model,y,x,new_sess)
    return ddlogβ,ddlogpy
end

"""
    compute_posteriors!(logβ, logpy, dlogβ, dlogpy, ddlogβ, ddlogpy, model, y, x, new_sess)

In place computation of posteriors, gradient, and hessians for model

"""

function compute_posteriors!(logβ,logpy,dlogβ,dlogpy,ddlogβ,ddlogpy,model::ModelDrift,y::AbstractVector{S},x,new_sess) where S <: Bool
    @unpack β,σ,σInit,σSess = model
    #@unpack β,βInit,σ,σInit = model


    ntrials = length(y)
    nfeats = size(x,1)

    if (logβ !== nothing) || (dlogβ !== nothing) || (ddlogβ !== nothing)
        invσ = fill_inverse_σ(σ,σInit,σSess,ntrials,new_sess)
        #invσ = fill_inverse_σ(σ,σInit,ntrials,new_sess)

        Δβ = D_v(β)
        # Δβ = D_v(β,βInit)

        if logβ !== nothing
            logβ .= (1/2) * (sum(log.(invσ)) - sum(Δβ .^ 2 .* invσ))[1]
        end

        if dlogβ !== nothing
            dlogβ .= -DT_v(invσ .* Δβ, nfeats)
        end

        if ddlogβ !== nothing
            ddlogβ .= -DT_Σ_D(invσ, nfeats)
        end

    end

    if (logpy !== nothing) || (dlogpy !== nothing) || (ddlogpy !== nothing)

        βx = vec(sum(β .* x, dims=1))

        if logpy !== nothing
            logpy .= sum(y .* βx .- logaddexp.(0, βx))  
        end


        if (dlogpy !== nothing) || (ddlogpy !== nothing)
            pL = 1 ./ (1 .+ exp.(-βx))

            if dlogpy !== nothing
                dlogpy .= vec(x .* permutedims(y - pL))
            end

            if ddlogpy !== nothing
                pL2 = (pL .^ 2 .- pL)
                ddlogpy .= sparse_log_hess(pL2, x, ntrials, nfeats)
            end

        end

    end

end
