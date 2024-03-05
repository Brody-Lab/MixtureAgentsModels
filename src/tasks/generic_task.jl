"""
    GenericData{T1,T2,T3,T4} <: RatData

Example behavioral data struct. Replace `Generic` with task name when creating for a specific task. Required fields are ones that must be included for models to run correctly. Optional fields can be added for agent compatibility. Check an agents `next_Q!` function to see what behavior data it requires.
(Behavior data is not limited to specifically "rat" data, but can be from any subject. But aren't we all just differently sized rats at the end of the day?)
See `tasks/twostep_task.jl` for a comprehensive example of how to create a data struct for a specific task.

Per julia guidelines, it's recommended to explicitly define the type of each field in the constructor. This ensures faster compilation times and faster performance. 

# Required Fields:
- `ntrials`: total number of trials
- `nfree`: total number of `free` trials (e.g. the choice was not forced, as indicated by `forced`). defaults to `ntrials`
- `choices`: (1=primary, 2=secondary) choices on each trial. this is what is being predicted by the model
- `new_sess`: boolean vector that is `true` if trial marks the start of a session
- `new_sess_free`: boolean vector that marks the first free choice at the start of a session. defaults to `new_sess`
- `forced`: boolean vector that is `true` if the choice was forced. these trials are excluded from choice likelihood estimation. defaults to `falses(ntrials)`

# Generated Fields:
- `sess_inds`: vector of vectors of trial indices for each session
- `sess_inds_free`: vector of vectors of free trial indices for each session

# Example Optional Fields:
- `rewards`: (1=reward, -1=omission) rewards on each trial
- `stim`: some stimulus on each trial, e.g. sound frequency, click rate, etc. If the stimulus is side selective, use positive values for the primary side and negative values for the secondary side

"""
@with_kw struct GenericData{I<:Int,VI<:AbstractVector{Int},VB<:AbstractVector{Bool},VA<:AbstractVector} <: RatData 
    ntrials::I
    nfree::I = ntrials
    choices::VI
    rewards::VI
    # stim::VI
    new_sess::VB
    new_sess_free::VB = new_sess
    forced::VB = falses(ntrials)
    sess_inds::VA = get_sess_inds(new_sess)
    sess_inds_free::VA = get_sess_inds(new_sess_free)
end

"""
    GenericData(data::D) where D <: Union{Dict,JSON3.Object}

Converts a dictionary or JSON object to a GenericData struct.
"""
function GenericData(data::D) where D <: Union{Dict,JSON3.Object}
    # remove "type" field if it exists; created when using `ratdata2dict`
    if "type" in keys(data)
        delete!(data,"type")
    end
    return GenericData(; data...)
end

"""
    GenericSim{T} <: SimOptions

Parameters for simulating task data. Required fields are used for generating sessions of fixed or variable length.
Add optional parameters specific to simulating task data

Required Fields:
- `nsess`: number of sessions
- `ntrials`: total number of trials 
- `mean_ntrials`: (overrides `ntrials`) mean number of trials/session if randomly drawing 
- `std_ntrials`: standard deviation of number of trials/session if randomly drawing. defaults to `Int(mean_ntrials/5)`
"""
@with_kw struct GenericSim{T} <: SimOptions where T
    nsess::T = 1
    ntrials::T = 1000
    mean_ntrials::T = 0
    std_ntrials::T = Int(mean_ntrials/5)
end

"""
    simulate_task(model, agents, sim_options, new_sess)

Task-specific function to simulate behavior data; to be called by `simulate_task(sim_options, model_options)` in `simulate_task.jl` 
Required for parameter recovery simulations
"""
function simulate_task(model::M,agents::Array{A},sim_options::GenericSim,new_sess::AbstractArray{Bool}) where {M <: MixtureAgentsModel, A <: Agent}
    @unpack ntrials = sim_options
    
    # code to simulate task data. 
    
    # return populated GenericData struct
    return data
end

"""
    load_EXAMPLE(file::String,...)

Custom function to load behaiovral data for EXAMPLE task from file. 

If it is a .mat file, you can use `matread` to load the data, and then transform it into variable types required by GenericData struct. 
You can also use `CSV.read` to load a .csv file into a DataFrame, and then convert it to a dictionary for GenericData struct.
See "data/example_task_data.csv" for an example of how to format a .csv file for GenericData struct, and "data/example_task_data.mat" for an example of how to format a .mat file for GenericData struct.
"""
function load_generic(file::String="data/example_task_data.csv")
    ext = split(file,'.')[end] # get file extension

    if ext == "mat"
        return load_generic_mat(file)

    elseif ext == "csv"
        return load_generic_csv(file)
    end

end

function load_generic_mat(file="data/example_task_data.mat")
        # code to load data from file
        matdata = matread(file)
        varname = collect(keys(matdata))[1] # this assumes there's only one variable in the .mat file
        matdata = matdata[varname]
        # package results into dictionary D for easy conversion to GenericData
        D = Dict{Symbol,Any}(
            :ntrials => length(matdata["choices"]), # number of trials
            :choices => vec(Int.(matdata["choices"])), # convert to int, make sure it's a vector
            :rewards => vec(Int.(matdata["rewards"])), # convert to int, make sure it's a vector
            # :stim => vec(Int.(matdata["stim"])), # convert to int, make sure it's a vector
            :new_sess => vec(matdata["new_sess"] .== 1)) # convert to boolean, make sure it's a vector
    
        # return GenericData struct
        return GenericData(D)
end

function load_generic_csv(file="data/example_task_data.csv")
    # load into dataframe
    df = CSV.read(file,DataFrame)
    # convert to dict
    # you may need to explicitly re-type things to avoid weirdly loaded types, like "String7" instead of "String". see "pclicks_task.jl" for an example
    # D = Dict{Symbol,Any}(pairs(eachcol(df)))
    D = Dict{Symbol,Any}(
        :choices=>Int.(df.choices),
        :rewards=>Int.(df.rewards),
        # :stim=>Int.(df.stim),
        :new_sess=>Bool.(df.new_sess)
    )

    D[:ntrials] = length(D[:choices])
    # D[:new_sess] = D[:new_sess] .== 1 # convert to boolean vector

    return GenericData(D)
end