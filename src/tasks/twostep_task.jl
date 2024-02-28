

"""
    TwoStepData{S,I,P,VB,VI,VF,VA} <: RatData where {S<:String,I<:Int,P<:Float64,VI<:AbstractVector{Int},VB<:AbstractVector{Bool},VF<:AbstractVector{Float64},VA<:AbstractVector}

Two-step task behavioral data for rats

Fields:
- `ratname`: name of rat
- `infusion`: type of infusion (e.g. "none", "saline", "muscimol")
- `ntrials`: total number of trials
- `nfree`: total number of `free` trials (e.g. the choice was not forced, as indicated by `forced`). Defaults to `ntrials`
- `choices`: (1=left, 2=right) first-step choices
- `nonchoices`: opposite index of choices
- `outcomes`: (1=left, 2=right) second-step outcomes
- `nonoutcomes`: opposite index of outcomes
- `rewards`: (1=reward, -1=omission) second-step rewards
- `trans_commons`: (1=common, 0=uncommon) boolean vector indicating whether or not the transition was common or rare
- `new_sess`: boolean vector that is `true` if trial marks the start of a session
- `new_sess_free`: boolean vector that marks the first free choice at the start of a session. Defaults to `new_sess`
- `forced`: boolean vector that is `true` if the first-step choice was forced (e.g. the rat was directed to choose a particular side)
- `p_congruent`: probability of a choice leading to its congruent outcome (e.g. left choice -> left outcome)
- `leftprobs`: boolean vector that is `true` if the left outcome is more probable than the right outcome
- `trial_start`: time of trial initialization
- `trial_end`: time of trial completion
- `sess_inds`: vector of session index ranges
- `sess_inds_free`: vector of session index ranges for free trials
"""

@with_kw struct TwoStepData{S<:String,VS<:Union{AbstractVector{String},Nothing},I<:Int,F<:Float64,VB<:AbstractVector{Bool},VI<:AbstractVector{Int},VF<:AbstractVector{Float64},VA<:AbstractVector} <: RatData 
    ratname::S
    infusion::S
    sessiondate::VS=nothing
    ntrials::I
    nfree::I = ntrials
    choices::VI
    nonchoices::VI
    outcomes::VI
    nonoutcomes::VI
    rewards::VI
    trans_commons::VB
    new_sess::VB
    new_sess_free::VB = new_sess
    forced::VB = falses(ntrials)
    p_congruent::F
    leftprobs::VB
    step1_times::VF
    step2_times::VF
    choice_times::VF
    outcome_times::VF
    sess_inds::VA = get_sess_inds(new_sess)
    sess_inds_free::VA = get_sess_inds(new_sess_free)
end

"""
    TwoStepSim

Parameters for simulating two-step task data

Fields:
- `nsess`: number of sessions
- `ntrials`: total number of trials 
- `mean_ntrials`: (overrides `ntrials`) mean number of trials/session if randomly drawing 
- `std_ntrials`: standard deviation of number of trials/session if randomly drawing
- `ntrials_for_flip`: minumum number of trials before reward flip can occur
- `flip_prob`: probability of reward flip
- `p_reward`: probability of reward on `good` side
- `p_congruent`: probability of a choice leading to its congruent outcome (e.g. left choice -> left outcome)

"""
@with_kw struct TwoStepSim{I<:Int,F<:Float64} <: SimOptions
    nsess::I = 1
    ntrials::I = 1000
    mean_ntrials::I = 0
    std_ntrials::I = Int(mean_ntrials/5)
    ntrials_for_flip::I = 10
    flip_prob::F = 0.02
    p_reward::F = 0.8
    p_congruent::F = 0.8
end

"""
    conversion functions for saving to and loading from dicts
"""
function TwoStepData(data::D) where D <: Union{Dict,JSON3.Object}
    return TwoStepData(; data...)
end


"""
    load_twostep(file, varname, rat, sessions)

Loads in MATLAB data from the two-step task and packages relevant parameters into a RatData struct.
"""
function load_twostep(file::String,rat::Any=nothing;infusion::Any=nothing,sessions::Any=nothing)
    matdata = matread(file)
    if length(keys(matdata)) > 1
        error("Unsupported .mat file; more than one variable in file.")
    else
        varname = collect(keys(matdata))[1]
    end
    if isnothing(infusion)
        if !isnothing(rat)
            if typeof(rat) == String
                rat_i = findall(map(d->d["ratname"],matdata[varname]) .== rat)
                if length(rat_i) < 1
                    error("No rat "*rat*" in dataset specified.")
                else
                    rat = rat_i[1]
                end
            end

            data = matdata[varname][rat]
        else
            data = matdata[varname]
        end
        infusion = "none"
    else
        if !isnothing(rat)
            tmp = matdata[varname][infusion][map(d->"ratname"∈keys(d),matdata[varname][infusion])]
            if typeof(rat) == String
                rat_i = findall(map(d->d["ratname"],tmp) .== rat)
                if length(rat_i) < 1
                    error("No rat "*rat*" in dataset specified.")
                else
                    rat = rat_i[1]
                end
            end
            data = matdata[varname][infusion][rat]
        else
            data = matdata[varname][infusion]
        end
        infusion = infusion[6:end]
    end

    #ntrials = Int.(data["nTrials"])
    # vector fields
    D = Dict{Any,Any}( 
        :choices => Int.(data["sides1"] .== "r") .+ 1,
        :nonchoices => Int.(data["sides1"] .== "l") .+ 1,
        :outcomes => Int.(data["sides2"] .== "r") .+ 1,
        :nonoutcomes => Int.(data["sides2"] .== "l") .+ 1,
        :rewards => vec(Int.(data["rewards"])) - vec(Int.((!=(1)).(data["rewards"]))),
        :trans_commons => vec(convert(BitArray,data["trans_common"])),
        :new_sess => vec(convert(BitArray,data["new_sess"])),
        :forced => vec(isnan.(data["better_choices"])),
        :leftprobs => vec(data["leftprobs"] .> 0.5),
        :step1_times => vec(data["c1_times"]),
        :step2_times => vec(data["c2_times"]),
        :choice_times => vec(data["s1_times"]),
        :outcome_times => vec(data["s2_times"]))

    viols = vec(!=(0).(data["viols"])) .| (data["sides2"].=="v") .| any.(eachrow(isnan.(hcat(D[:step1_times],D[:choice_times],D[:step2_times],D[:outcome_times]))))
    viol_starts = findall(viols .& D[:new_sess])
    for i = viol_starts
        first_good = findfirst(.!viols[i[1]+1:end])
        D[:new_sess][i[1]+first_good] = true
    end

    map(field->remove_viols!(D[field],viols),collect(keys(D)))
    if !isnothing(sessions)
        sess_use = use_sessions(sessions,D[:new_sess],get_sess_inds(D[:new_sess]))
        map(field->trim_sessions!(D[field],sess_use),collect(keys(D)))
    end

    if haskey(data,"sessiondate")
        D[:sessiondate] = string.(vec(data["sessiondate"]))
        if !isnothing(sessions)
            D[:sessiondate] = D[:sessiondate][sessions]
        end
    end

    # get ind of first free trial in a each session
    new_sess_free = copy(D[:new_sess])
    forced_starts = findall(D[:forced] .& D[:new_sess])
    for i = forced_starts
        first_free = findfirst(.!D[:forced][i[1]+1:end])
        new_sess_free[i[1]+first_free] = true
    end
    deleteat!(new_sess_free,findall(D[:forced]))
    D[:new_sess_free] = new_sess_free

    # non-vector fields
    D[:ratname] = data["ratname"]
    D[:infusion] = infusion
    D[:p_congruent] = data["p_congruent"]
    D[:ntrials] = length(D[:choices])
    D[:nfree] = sum(D[:forced].==0)


    return TwoStepData(D)

end

"""
    not_sessions(sess_use)

Inverted logical indices of `sess_use`. For backwards compatibility with julia < v"1.7"
"""
function not_sessions(sess_use)
    return sess_use .== 0
end

"""
    remove_viols!(field,viols)

Remove entries of `field` at `viols`
"""
function remove_viols!(field::AbstractVector{T},viols) where T
    if (T <: Bool) && VERSION < v"1.7"
        deleteat!(field,findall(viols))
    else
        deleteat!(field,viols)
    end
end

function remove_viols!(field::AbstractMatrix{T},viols) where T
    field = deleterows(field,viols)
end

"""
    simulate(model, agents, sim_options, new_sess)

Simulates two-step data according to sim_options, generated by model specified by model_options

Optional arguments:
- `return_z`: (default = false) return generated latent state `z`
- `seed`: (default = nothing) set seed for random number generator
"""
function simulate(model::M,agents::Array{A},sim_options::TwoStepSim,new_sess::AbstractArray{Bool};return_z=false,seed=nothing) where {M <: MixtureAgentsModel, A <: Agent}
    @unpack ntrials, ntrials_for_flip, flip_prob, p_reward, p_congruent = sim_options
    
    reward_probs = zeros(2,ntrials)

    if !isnothing(seed)
        Random.seed!(seed)
    end

    # initial reward side
    if rand() < 0.5
        p_reward_left = p_reward
    else
        p_reward_left = 1 - p_reward
    end

    # session rewards
    flip_counter = 0
    for trial = 1:ntrials
        if new_sess[trial]
            flip_counter = 0
        end
        if flip_counter > ntrials_for_flip
            if rand() < flip_prob
                p_reward_left = 1 - p_reward_left
            end
        end
        reward_probs[1,trial] = p_reward_left
        reward_probs[2,trial] = 1 - p_reward_left
        flip_counter += 1
    end

    return generate_data(agents,model,ntrials,reward_probs,p_congruent,new_sess;return_z=return_z,seed=seed)
end

"""
    generate_data(agents::Array{A},model::M,ntrials::Int,reward_probs::Array{Float64},p_congruent::Float64,new_sess::AbstractArray{Bool};return_z=false,seed=nothing) where {A <: Agent, M<:MixtureAgentsModel}

Generates task parameters required for simulated TwoStepData
"""
function generate_data(agents::Array{A},model::M,ntrials::Int,reward_probs::Array{Float64},p_congruent::Float64,new_sess::AbstractArray{Bool};return_z=false,seed=nothing) where {A <: Agent, M<:MixtureAgentsModel}

    ratname = "sim"
    infusion = "none"
    choices = Array{Int}(undef,ntrials)
    nonchoices = Array{Int}(undef,ntrials)
    outcomes = Array{Int}(undef,ntrials)
    nonoutcomes = Array{Int}(undef,ntrials)
    rewards = Array{Int}(undef,ntrials)
    trans_commons = BitVector(undef,ntrials)
    leftprobs = reward_probs[1,:] .> 0.5
    step1_times = Array{Float64}(undef,ntrials)
    step2_times = Array{Float64}(undef,ntrials)
    choice_times = Array{Float64}(undef,ntrials)
    outcome_times = Array{Float64}(undef,ntrials)

    data = TwoStepData(ratname=ratname,infusion=infusion,ntrials=ntrials,choices=choices,nonchoices=nonchoices,outcomes=outcomes,nonoutcomes=nonoutcomes,rewards=rewards,trans_commons=trans_commons,new_sess=new_sess,p_congruent=p_congruent,step1_times=step1_times,step2_times=step2_times,leftprobs=leftprobs,choice_times=choice_times,outcome_times=outcome_times)

    if !isnothing(seed)
        Random.seed!(seed)
    end

    # x = zeros(length(agents[:,1]),ntrials)
    # Q = SizedMatrix{length(agents[:,1])}(init_Q(agents[:,1]))
    x = zeros(length(agents),ntrials)
    Q = SizedMatrix{length(agents)}(init_Q(agents))
    z = zeros(Int,ntrials)
    pz = 1
    for t = 1:ntrials  
        # get x
        if new_sess[t]
            # Q .= init_Q(agents[:,1])
            Q .= init_Q(agents)
        end

        x[:,t] = Q[:,1] .- Q[:,2]
        
        # determine choice and outcome probability
        pL,pz,z[t] = sim_choice_prob(model,x,t,pz,new_sess)
        if rand() < pL
            choices[t] = 1
            nonchoices[t] = 2
            outcome_prob = p_congruent
        else
            choices[t] = 2
            nonchoices[t] = 1
            outcome_prob = 1 - p_congruent
        end

        # determine outcome
        if rand() < outcome_prob
            outcomes[t] = 1
            nonoutcomes[t] = 2
        else
            outcomes[t] = 2
            nonoutcomes[t] = 1
        end

        # determine reward
        if rand() < reward_probs[outcomes[t],t]
            rewards[t] = 1
        else
            rewards[t] = -1
        end

        # transition
        if (choices[t] == outcomes[t] && p_congruent > 0.5) || (choices[t] != outcomes[t] && p_congruent < 0.5)
            trans_commons[t] = true
        else
            trans_commons[t] = false
        end

        # update values
        # next_Q!(Q,agents[:,z[t]],data,t)
        next_Q!(Q,agents,data,t)
    end
    if return_z
        return data,z
    else
        return data
    end
end