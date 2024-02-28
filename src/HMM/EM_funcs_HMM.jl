"""
    optimize(model::ModelHMM, agents::AbstractArray{A}, y, x, data::D, model_options::ModelOptionsHMM, agent_options::AgentOptions)

Single EM iteration to optimize model and agent parameters
"""
function optimize(model::ModelHMM,agents::AbstractArray{A},y,x,data::D,model_options::ModelOptionsHMM,agent_options::AgentOptions,data_test::Union{D,Nothing}=nothing) where {A<:Agent,D<:RatData}
    @unpack new_sess_free = data
    @unpack nstates,βprior,α_π,α_A= model_options
    @unpack fit_inds,fit_priors = agent_options
    γs,ξs,_ = compute_posteriors(model,y,x,new_sess_free)

    if nstates > 1
        model = update(model,γs,ξs,new_sess_free,α_π,α_A)
    end

    @unpack β = model
    if !isnothing(fit_inds)
        params = get_params(agents,agent_options)
        scale = [0,get_scales(agents,agent_options)]
        # scale = [0,1]
        vars = [β,params]
        v0 = vectorize(vars,scale)
        if !isnothing(fit_priors)
            param_priors = get_priors(agents,agent_options)
        else
            param_priors = nothing
        end
    else
        scale = [0]
        vars = [β]
        v0 = Array(vec(β))
        param_priors = nothing
    end

    if βprior
        # L2θ = get_param(agents,:θL2) 
        β_priors = permutedims(reduce(hcat,get_param(agents,:βprior)))
    else
        # L2θ = nothing
        β_priors = nothing
    end

    agents_fit = deepcopy(agents)
    #x_fit = copy(x)
    ℓℓ(v) = negloglikelihood(v,vars,scale,y,x,γs,agents_fit,agent_options,data,β_priors,param_priors)
    obj = OnceDifferentiable(ℓℓ, v0; autodiff=:forward)
    res = Optim.optimize(obj, v0, LBFGS(linesearch = LineSearches.BackTracking()))
    v = Optim.minimizer(res)

    if !isnothing(fit_inds)
        β,params = extract(v,vars,scale)
        update!(x,agents_fit,params,agent_options,data)
        # update!(x,agents_fit[fit_inds],fit_inds,data)
    else
        β = SizedMatrix{size(vars[1])...}(v)
    end
    model_fit = Accessors.set(model,opcompose(PropertyLens(:β)),β)
    # ecll = ex_loglikelihood(model_fit,agents_fit,model_options,agent_options,data)
    ll = marginal_likelihood(model_fit,agents_fit,model_options,agent_options,data)
    # _,_,ll = compute_posteriors(model_fit,y,x,new_sess_free)


    if !isnothing(data_test)
        _,_,ll_test = compute_posteriors(model_fit,agents_fit,data_test)
    else
        ll_test = 0.
    end

    return model_fit,agents_fit,ll,ll_test
end

"""
    negloglikelihood(v, vars, scale, y, x, γs, agents, options, data)

Computes the negative log-likelihood of the MoA weights for gradient descent.
"""
function negloglikelihood(v::Vector{T},vars::AbstractVector,scale::AbstractVector,y::BitVector,x::Array{Float64},γs::Array{Float64},agents::Array{A},options::AgentOptions,data::RatData,β_priors::Union{AbstractArray,Nothing},param_priors::Union{AbstractArray,Nothing}) where {T <: Union{Float64,ForwardDiff.Dual}, A <: Agent}
    @unpack fit_inds = options
    agents_tmp = deepcopy(agents)

    if !isnothing(fit_inds)
        x = convert(Array{T},x)
        β,params = extract(v,vars,scale)
        update!(x,agents_tmp,params,options,data)

        return -ex_loglikelihood(γs,y,x,β,β_priors,params,param_priors)
    else
        β = SizedMatrix{size(vars[1])...}(v)

        return -ex_loglikelihood(γs,y,x,β,β_priors,nothing,nothing)
    end
end

function ex_loglikelihood(γs::Array,y::BitVector,x::Array,β::AbstractArray,β_priors::Union{AbstractArray,Nothing},params::Union{AbstractVector,Nothing},param_priors::Union{AbstractArray,Nothing})
    βx = βT_x(β,x)
    f = sum(@. γs * (y' * βx - logaddexp.(0, βx)))
    
    if !isnothing(β_priors)
        f += sum(logprior.(β_priors,β))
    end

    if !isnothing(param_priors)
        f += sum(logprior.(param_priors,params))
    end
    
    return f
end

function ex_loglikelihood(model::ModelHMM,agents::Array{A},model_ops::ModelOptionsHMM,agent_ops::AgentOptions,data::D) where {D <: RatData, A <: Agent}
    @unpack βprior,α_π,α_A = model_ops
    @unpack fit_inds,fit_priors = agent_ops
    if !isnothing(fit_inds)
        params = get_params(agents,agent_ops)
        if !isnothing(fit_priors)
            param_priors = get_priors(agents,agent_ops)
        else
            param_priors = nothing
        end
    else
        params = nothing
        param_priors = nothing
    end
    if βprior
        β_priors = permutedims(reduce(hcat,get_param(agents,:βprior)))
    else
        β_priors = nothing
    end

    y,x = initialize(agents,data)
    γs,ξs,_ = compute_posteriors(model,y,x,data.new_sess_free)
    f = ex_loglikelihood(γs,y,x,model.β,β_priors,params,param_priors)

    f += nansum(γs[:,data.new_sess_free] .* log.(model.π)) + logprior(Dirichlet(α_π),model.π)
    f += nansum(ξs .* log.(model.A)) + logprior(Dirichlet(α_A[:]),model.A[:])

    return f
end

"""
    update(model, γs, ξs, new_sess)

Optimizes the transition matrix and initial state probabilities using data
posteriors.
"""
function update(model::ModelHMM,γs,ξs,new_sess,α_π,α_A)

    πi = max.(dropdims(sum(γs[:,new_sess],dims=2),dims=2) + α_π .- 1,0)
    πi ./= sum(πi)

    #A = dropdims(sum(ξs,dims=3),dims=3){I<:Int,Q<:AbstractVector,L<:Real,B<:AbstractVector,C<:Symbol,S<:Symbol}
    A = max.(ξs + α_A .- 1,0)
    A ./= sum(A,dims=2)

    model = Accessors.set(model,opcompose(PropertyLens(:π)),πi)
    model = Accessors.set(model,opcompose(PropertyLens(:A)),A)
    #model = Parameters.reconstruct(model,π=π,A=A)
    return model
end

"""
    compute_posteriors(model, y, x, new_sess)

Computes conditional latent state posterior probabilities and joint latent state
posterior probabilites using the forward-backward algorithm, as well as the
marginal likelihood.
"""
function compute_posteriors(model::ModelHMM,y::AbstractVector{Bool},x::Array{Float64},new_sess::AbstractVector{Bool};sum_ξ=true)
    @unpack β,π,A = model

    nstates = size(β,2)
    ntrials = length(y)
    pL = logistic.(βT_x(β,x))
    y = permutedims(y)
    py_z = @. y * pL + (1 - y) * (1 - pL)

    αs = Array{Float64}(undef,nstates,ntrials)
    c = Array{Float64}(undef,ntrials)
    for t = 1:ntrials
        if new_sess[t]
            αs[:,t] = π .* py_z[:,t]
        else
            αs[:,t] = py_z[:,t] .* (A' * αs[:,t-1])
        end

        c[t] = sum(αs[:,t])
        αs[:,t] = αs[:,t] ./ c[t]
    end
    ll = 0.
    try
        ll = sum(log.(c))
    catch
        println(model.β)
        println(model.A)
        println(model.π)
        ll = sum(log.(c))
    end

    βs = Array{Float64}(undef,nstates,ntrials)
    βs[:,end] = ones(nstates)

    for t = ntrials-1:-1:1
        if new_sess[t+1]
            βs[:,t] = ones(nstates)
        else
            βs[:,t] = A * (βs[:,t+1] .* py_z[:,t+1])          # equation 13.38
            βs[:,t] = βs[:,t] ./ c[t+1]
        end
    end

    γs = αs .* βs

    ts = collect(1:ntrials)
    deleteat!(ts,new_sess)

    # ξs = ((αs[:, ts .- 1] ./ permutedims(c[ts])) * permutedims(py_z[:,ts].*βs[:,ts])) .* A

    if sum_ξ
        ξs = ((αs[:, ts .- 1] ./ permutedims(c[ts])) * permutedims(py_z[:,ts].*βs[:,ts])) .* A
    else
        ξs = Array{Float64}(undef,nstates,nstates,length(ts))
        for (i,t) in enumerate(ts)
            ξs[:,:,i] = ((αs[:,t-1] ./ c[t]) * (py_z[:,t] .* βs[:,t])') .* A
        end
    end

    return γs,ξs,ll
end

function compute_posteriors(model::ModelHMM,agents::Array{A},data::D;sum_ξ=true) where {D <: RatData, A <: Agent}
    @unpack new_sess_free = data
    y,x = initialize(data,agents)
    return compute_posteriors(model,y,x,new_sess_free;sum_ξ=sum_ξ)
end



function marginal_likelihood(model::ModelHMM,agents::Array{A},model_ops::ModelOptionsHMM,agent_ops::AgentOptions,data::D) where {D <: RatData, A <: Agent}
    @unpack βprior,α_π,α_A = model_ops
    @unpack fit_inds,fit_priors = agent_ops
    _,_,ll = compute_posteriors(model,agents,data)

    if !isnothing(fit_priors)
        params = get_params(agents,agent_ops)
        param_priors = get_priors(agents,agent_ops)
        ll += sum(logprior.(param_priors,params))
    end

    if βprior
        β_priors = permutedims(reduce(hcat,get_param(agents,:βprior)))
        ll += sum(logprior.(β_priors,model.β))
    end

    ll += nansum(logprior(Dirichlet(α_π),model.π))
    ll += nansum(logprior(Dirichlet(α_A[:]),model.A[:]))

    return ll
end


function choice_likelihood(model::ModelHMM,agents::Array{A},data::D) where {D <: RatData, A <: Agent}
    @unpack β = model
    y,x = initialize(data,agents)
    γs,_,ll = compute_posteriors(model,y,x,data.new_sess_free)

    pL = logistic.(βT_x(β,x))
    yT = permutedims(y)
    py_z = yT .* pL + (1 .- yT) .* (1 .- pL)

    py = sum(γs .* py_z,dims=1)[:]
    pL = sum(γs .* pL,dims=1)[:]

    return py,py_z,pL,γs,ll   
end
