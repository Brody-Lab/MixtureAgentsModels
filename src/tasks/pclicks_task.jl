"""
    pclicksdata{S<:String,I<:Int,VS<:AbstractVector{String},VI<:AbstractVector{Int},VB<:AbstractVector{Bool},VF<:AbstractVector{Float64},VA<:AbstractVector} <: RatData EXAMPLEdata{T1,T2,T3,T4} <: RatData

Data struct for Poisson Clicks task in rats. Also includes fields for 2-armed bandit/reward reversal task that can be switched to during training.

# Required Fields:
- `ntrials`: total number of trials
- `task`: active task, either "delta clicks" or "reward_reversal"
- `nfree`: total number of `free` trials (e.g. the choice was not forced, as indicated by `forced`). defaults to `ntrials`
- `choices`: (1=right, 2=left) choices on each trial. 
- `rewards`: (1=reward, -1=omission) rewards on each trial
- `leftprobs`: probability of reward on left side on each trial during reward reversal task
- `rightprobs`: probability of reward on right side on each trial during reward reversal task
- `nleftclicks`: number of left clicks on each trial 
- `nrightclicks`: number of right clicks on each trial 
- `new_sess`: boolean vector that is `true` if trial marks the start of a session
- `new_sess_free`: boolean vector that marks the first free choice at the start of a session. defaults to `new_sess`
- `forced`: boolean vector that is `true` if the choice was forced. these trials are excluded from choice likelihood estimation. defaults to `falses(ntrials)`

# Generated Fields:
- `leftclicksnorm`: left clicks normalized by maximum number of left clicks across all trials. 
- `rightclicksnorm`: right clicks normalized by maximum number of right clicks across all trials.
- `sess_inds`: vector of vectors of trial indices for each session
- `sess_inds_free`: vector of vectors of free trial indices for each session

"""
@with_kw struct pclicksdata{S<:String,I<:Int,VS<:AbstractVector{String},VI<:AbstractVector{Int},VB<:AbstractVector{Bool},VF<:AbstractVector{Float64},VA<:AbstractVector} <: RatData 
    ratname::S
    task::VS
    ntrials::I
    nfree::I = ntrials
    choices::VI
    rewards::VI
    leftprobs::VF
    rightprobs::VF
    nleftclicks::VI
    nrightclicks::VI
    zleftclicks::VF = (nleftclicks .- mean(vcat(nleftclicks,nrightclicks))) ./ std(nrightclicks .- nleftclicks)
    zrightclicks::VF = (nrightclicks .- mean(vcat(nleftclicks,nrightclicks))) ./ std(nrightclicks .- nleftclicks)
    new_sess::VB
    new_sess_free::VB = new_sess
    forced::VB = falses(ntrials)
    sess_inds::VA = get_sess_inds(new_sess)
    sess_inds_free::VA = get_sess_inds(new_sess_free)
end

"""
    pclicksdata(data::D) where D <: Union{Dict,JSON3.Object}

Converts a dictionary or JSON object to a pclicksdata struct.
"""
function pclicksdata(data::D) where D <: Union{Dict,JSON3.Object}
    # remove "type" field if it exists; created when using `ratdata2dict`
    if "type" in keys(data)
        delete!(data,"type")
    end
    return pclicksdata(; data...)
end

# """
#     pclickssim{T} <: SimOptions

# Parameters for simulating task data. Required fields are used for generating sessions of fixed or variable length.
# Add optional parameters specific to simulating task data

# Required Fields:
# - `nsess`: number of sessions
# - `ntrials`: total number of trials 
# - `mean_ntrials`: (overrides `ntrials`) mean number of trials/session if randomly drawing 
# - `std_ntrials`: standard deviation of number of trials/session if randomly drawing. defaults to `Int(mean_ntrials/5)`
# """
# @with_kw struct pclickssim{T} <: SimOptions where T
#     nsess::T = 1
#     ntrials::T = 1000
#     mean_ntrials::T = 0
#     std_ntrials::T = Int(mean_ntrials/5)
# end

# """
#     simulate_task(model, agents, sim_options, new_sess)

# Task-specific function to simulate behavior data; to be called by `simulate_task(sim_options, model_options)` in `simulate_task.jl` 
# Required for parameter recovery simulations
# """
# function simulate_task(model::M,agents::Array{A},sim_options::pclickssim,new_sess::AbstractArray{Bool}) where {M <:MixtureAgentsModel, A <: Agent}
#     @unpack ntrials = sim_options
    
#     # code to simulate task data. 
    
#     # return populated EXAMPLEdata struct
#     return data
# end

"""
    load_pclicks(file::String,...)

Custom function to load behavioral data for pclicks task. Currently supports .csv files saved according to:
https://github.com/Brody-Lab/UberPhys/blob/main/thomas/analyses/analysis_2023_12_12a_behavioraldata/2023_12_12a.md

If you want to load data from a different file type, you can add a new function.
"""
function load_pclicks(file::String,rat::String;sessions::Any=nothing)
    ext = split(file,'.')[end] # get file extension

    if ext == "csv"
        return load_pclicks_csv(file,rat;sessions=sessions)
    else
        error("File type not supported")
    end

end

# function load_EXAMPLE_mat(file="data/example_task_data.mat")
#         # code to load data from file
#         matdata = matread(file)
#         varname = collect(keys(matdata))[1] # this assumes there's only one variable in the .mat file
#         matdata = matdata[varname]
#         # package results into dictionary D for easy conversion to EXAMPLEdata
#         D = Dict{Symbol,Any}(
#             :ntrials => length(matdata["choices"]), # number of trials
#             :choices => vec(Int.(matdata["choices"])), # convert to int, make sure it's a vector
#             :rewards => vec(Int.(matdata["rewards"])), # convert to int, make sure it's a vector
#             :new_sess => vec(matdata["new_sess"] .== 1), # convert to boolean, make sure it's a vector
#             :stim => vec(Int.(matdata["stim"]))) # convert to int, make sure it's a vector
    
#         # return EXAMPLEdata struct
#         return EXAMPLEdata(D)
# end
"""
    load_pclicks_csv(file::String,rat::String)

Type-specific function to load behavioral data for pclicks task from .csv file saved according to:
    https://github.com/Brody-Lab/UberPhys/blob/main/thomas/analyses/analysis_2023_12_12a_behavioraldata/2023_12_12a.md

If the page above doesn't exist, have you considered that it just doesn't exist for YOU?! Or ask Thomas
"""
function load_pclicks_csv(file::String,rat::String;sessions::Any=nothing)
    # load into dataframe
    df_full = CSV.read(file,DataFrame)
    df = groupby(df_full,:ratname)[(rat,)]
    # convert to dict
    rewards = df.reward
    rewards[rewards .== 0] .= -1
    new_sess = vcat(1,df.sessid[2:end] .!= df.sessid[1:end-1])
    # Everything is explicitly re-typed to avoid weirdly loaded types, like "String7" instead of "String"
    D = Dict{Symbol,Any}(
        :choices=>Int.((df.pokedR .== 0) .+ 1),
        :task=>String.(df.reward_type),
        :rewards=>Int.(rewards),
        :leftprobs=>Float64.(df.RR_left_reward_prob),
        :rightprobs=>Float64.(df.RR_right_reward_prob),
        :nleftclicks=>Int.(df.nleftclicks),
        :nrightclicks=>Int.(df.nrightclicks),
        :new_sess=>Bool.(new_sess)
    )

    if !isnothing(sessions)
        sess_use = use_sessions(sessions,D[:new_sess],get_sess_inds(D[:new_sess]))
        map(field->trim_sessions!(D[field],sess_use),collect(keys(D)))
    end

    D[:ratname] = String(df.ratname[1])
    D[:ntrials] = length(D[:new_sess])

    return pclicksdata(D)
end