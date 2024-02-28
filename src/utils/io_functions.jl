"""
    savevars(fname;var1,var2,...,varN)

Function to save arbitrary variable to file. Save format determined by extension in fname. Options are ".jld2", ".json", ".mat"
WARNING: matrices saved as .json will be flattened. save the original shape for reshaping if desired
"""
function savevars(fname;kwargs...) 
    file_parts = splitext(fname)
    if file_parts[2] == ".jld2"
        jldsave(fname;kwargs...)    

    elseif file_parts[2] == ".json"
        open(fname, "w") do io
            JSON3.write(io, Dict(kwargs))
        end

    elseif file_parts[2] == ".mat"
        k_dict = Dict(kwargs)
        k_dict_str = Dict(string(key)=>get(k_dict,key,nothing) for key in keys(k_dict))
        matwrite(fname,k_dict_str)
    end
end

"""
    loadvars(fname)

Function to load variables from file. Can only load files saved as ".jld2", ".json", or ".mat"
WARNING: if matrices were saved in a .json dict, they were flattened. unless the original shape was saved, there is currently no wait to reshape
"""
function loadvars(fname)
    file_parts = splitext(fname)

    if file_parts[2] == ".jld2"
        return jldopen(fname)    

    elseif file_parts[2] == ".json"
        return JSON3.read(read(fname, String))

    elseif file_parts[2] == ".mat"
        return matread(fname)

    else
        error("Unsupported file type "*file_parts[2])

    end

end

"""
    savefit(fname,model::S,model_ops::O,agents::AbstractArray{A},agent_ops::AgentOptions,data::D,ll::Float64=0.) where {S <: MixtureAgentsModel, O <: ModelOptions, A <: Agent, D<:RatData}

Function to save file model, model options, agents, agent options, data, and (optionally) log-likelihood of a model fit.
File type based on extension in fname. Types: ".jld2", ".json", ".mat"
"""
function savefit(fname,model::S,model_ops::O,agents::AbstractArray{A},agent_ops::AgentOptions,data::D,ll::Float64=0.) where {S <: MixtureAgentsModel, O <: ModelOptions, A <: Agent, D<:RatData}
    file_parts = splitext(fname)
    if file_parts[2] == ".jld2"
        jldsave(fname;model,model_ops,agents,agent_ops,data,ll)

    else
        model_dict = model2dict(model,model_ops)
        agents_dict = agents2dict(agents,agent_ops)
        data_dict = ratdata2dict(data)
        model_fit = Dict("model"=>model_dict,"agents"=>agents_dict,"ratdata"=>data_dict,"ll"=>ll)

        if file_parts[2] == ".json"
            open(fname, "w") do io
                JSON3.write(io, model_fit)
            end

        elseif file_parts[2] == ".mat"
            matwrite(fname,model_fit)

        else
            error("Unsupported file type "*file_parts[2])
        end
    end
end


"""
    loadfit(fname)

Function to load file containing model fit parameters
"""
function loadfit(fname)
    file_parts = splitext(fname)
    if file_parts[2] == ".jld2"
        model_fit = jldopen(fname)
        return model_fit["model"],model_fit["model_ops"],model_fit["agents"],model_fit["agent_ops"],model_fit["data"],model_fit["ll"]
    else
        if file_parts[2] == ".json"
            model_fit = JSON3.read(read(fname, String))      
        elseif file_parts[2] == ".mat"
            model_fit = matread(fname)
        else
            error("Unsupported file type "*file_parts[2])
        end
        model,model_ops = dict2model(model_fit["model"])
        agents,agent_ops = dict2agents(model_fit["agents"])
        data = dict2ratdata(model_fit["ratdata"])
        if !haskey(model_fit,"ll")
            _,_,model_fit["ll"] = compute_posteriors(model,agents,data)
        end
        return model,model_ops,agents,agent_ops,data,model_fit["ll"]
    end
end

"""
    make_fname(model_options::ModelOptions,agent_options::AgentOptions;ext=".jld2",nfold=nothing,num=nothing,iter=nothing)

Function to make filename for saving model fit parameters
"""
function make_fname(model_options::M,agent_options::AgentOptions;ext=".mat",rat=nothing,nfold=nothing,sim=nothing,num=nothing,iter=nothing,infusion=nothing) where M <: ModelOptions
    @unpack agents,symb_inds = agent_options
    
    header = fname_header(model_options)
    if !isnothing(rat)
        header = rat*"_"*header
    end

    alist = agents2list(agents)
    fname = header*"_"*alist

    if !isnothing(symb_inds)
        nparams = length(symb_inds)
        nparam_states = size(agents,2)
        param = string(nparams,"X",nparam_states,"param")
        fname *= "_"*param
    end

    if !isnothing(infusion)
        fname *= string("_",infusion)
    end

    if !isnothing(nfold)
        fname *= string("_",nfold,"fold")
    end
    if !isnothing(sim)
        fname *= string("_sim",sim)
    end
    if !isnothing(num)
        fname *= string("_n",num)
    end
    if !isnothing(iter)
        fname *= string("_i",iter)
    end

    return fname*ext
end

function fname_header(options::ModelOptionsHMM)
    @unpack nstates = options
    return string(nstates,"stateHMM")
end

function fname_header(options::ModelOptionsDrift)
    return "drift"
end


