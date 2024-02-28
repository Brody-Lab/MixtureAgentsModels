"""
    cross_validation(data::RatData,model_options::O,agent_options::AgentOptions;nfold::Int=3,method="ordered",seed::Int=0) where O <: ModelOptions

Performs nfold or leave-n-out cross-validation of model specified by `model_options` and `agent_options` on `data`. Returns average cross-validated model fit and average log-likelihood of training and test sets.

# Required Inputs:
- `data::RatData`: behavioral data struct
- `model_options::ModelOptions`: model options struct
- `agent_options::AgentOptions`: agent options struct

# Optional Arguments:
- `type::String`: type of cross-validation to perform. Options are "nfold" (default) or "leave_out"
- `nfold::Int`: number of folds to use (default = 3)
- `nout::Int`: number of sessions to leave out (default = 3)
- `method::String`: method for splitting data into folds. Options are "stride" (default) or "random"
- `seed::Int`: random seed for random splitting method
- `save_vars::Bool`: whether to save cross-validation results to file (default = false)
- `save_path::String`: path to save results (default to working directory)
- `fname::String`: filename to save results (default to generated filename from function `make_fname()`)
- `save_fold::Bool`: whether to save results for each fold (default = false)
- `rm_fold::Bool`: whether to remove results for each fold after completing all folds (default = false)
- `overwrite::Bool`: whether to overwrite existing results (default = false)
- `save_tmp::Bool`: whether to save temporary files for each fold's model fits (default = false)
- `rm_tmp::Bool`: whether to remove temporary files after completing fold (default = true)
"""
function cross_validate(data::RatData,model_options::O,agent_options::AgentOptions;type::String="nfold",nfold::Int=3,nout::Int=3,method::String="stride",seed::Int=0,save_vars::Bool=false,save_path=nothing,fname=nothing,save_fold::Bool=false,rm_fold::Bool=false,overwrite=false,save_tmp=false,rm_tmp=true) where O <: ModelOptions
    if save_vars
        if isnothing(save_path)
            save_path = pwd()
        elseif !isdir(save_path)
            mkpath(save_path)
        end
        if !save_fold
            save_tmp = false
        end
        if isnothing(fname)
            fname = make_fname(model_options,agent_options,nfold=nfold)
        end
        fpath = joinpath(save_path,fname)
        if isfile(fpath) & !overwrite
            println(fname," already created, loading results")
            vars = loadvars(fpath)
            @unpack ll_train,ll_test,model,agents = vars
            return dict2model(model)[1],dict2agents(agents)[1],ll_train,ll_test,nothing,nothing
        end
    end

    if type=="nfold"
        train_set,test_set = nfold_sessions(data;method=method,nfold=nfold,seed=seed)
    elseif type=="leave_out"
        train_set,test_set = leave_out_sessions(data;method=method,nout=nout,seed=seed)
        if nfold > length(train_set)
            nfold = length(train_set)
        end
    else
        error("unknown cross-validation type")
    end

    if seed > 0
        Random.seed!(seed)
    end

    n_train_trials = zeros(Int64,nfold)
    n_test_trials = zeros(Int64,nfold)
    n_ll_train = zeros(Float64,nfold)
    n_ll_test = zeros(Float64,nfold)
    model_train = Array{modeltype(model_options)}(undef,nfold)
    agents_train = Array{Array{Agent}}(undef,nfold)

    # Threads.@threads 
    for n = 1:nfold

        if save_vars & save_fold
            fname_n = fname[1:end-4]*"_n"*string(n)*fname[end-3:end]
            fpath_n = joinpath(save_path,fname_n)
            if isfile(fpath_n) & !overwrite
                println(fname_n," already created, loading results")
                vars = loadvars(fpath_n)
                n_ll_train[n] = vars["ll_train"]
                n_train_trials[n] = vars["train_trials"]
                n_ll_test[n] = vars["ll_test"]
                n_test_trials[n] = vars["test_trials"]
                model_train[n],_ = dict2model(vars["model"])
                agents_train[n],_ = dict2agents(vars["agents"])
                continue
            end
            if save_tmp
                fname_tmp = fname_n[1:end-4]*"_tmp"*fname_n[end-3:end]
            else
                fname_tmp = nothing
            end
        else
            fname_tmp = nothing
        end

        data_train,data_test = split_data(data,train_set[n],test_set[n])
        
        n_train_trials[n] = length(data_train.new_sess_free)
        n_test_trials[n] = length(data_test.new_sess_free)
        model_train[n],agents_train[n],n_ll_train[n] = optimize(data_train,model_options,agent_options;init_hypers=true,seed=seed,save_model=save_tmp,save_iter=save_tmp,save_path=save_path,fname=fname_tmp,overwrite=overwrite,disp_iter=10)
        _,_,n_ll_test[n] = compute_posteriors(model_train[n],agents_train[n],data_test)

        if save_vars & save_fold
            savevars(fpath_n;n,ll_train=n_ll_train[n],train_trials=n_train_trials[n],ll_test=n_ll_test[n],test_trials=n_test_trials[n],model=model2dict(model_train[n],model_options),agents=agents2dict(agents_train[n],agent_options),ratname=data.ratname,nfold,method,seed)
        end
        # remove temporary files
        if save_tmp & rm_tmp
            tmp_files = filter(x->contains(x,fname_tmp[1:end-4]),readdir(save_path,join=true))
            rm.(tmp_files)
        end


        GC.gc()
    end

    ll_train = exp(sum(n_ll_train) / sum(n_train_trials))
    ll_test = exp(sum(n_ll_test) / sum(n_test_trials))
    ll_train_n = exp.(n_ll_train ./ n_train_trials)
    ll_test_n = exp.(n_ll_test ./ n_test_trials)

    model = model_mean(model_train)
    agents = agents_mean(agents_train,agent_options)
    if save_fold & rm_fold
        fname_fold = fname[1:end-4]*r"_n\d"
        fold_files = filter(x->contains(x,fname_fold),readdir(save_path,join=true))
        rm.(fold_files)
    end

    if save_vars
        savevars(fpath;ll_train,ll_test,model=model2dict(model,model_options),agents=agents2dict(agents,agent_options),ratname=data.ratname,nfold,method,seed)
    end

    return model,agents,ll_train,ll_test,ll_train_n,ll_test_n

end

"""
    trim_data(data_in::D,sess_set) where D <: RatData

Trims all fields in `data_in` to only include sessions specified by `sess_set`
Splits fields by dependence on `new_sess` and `new_sess_free`
"""  
function trim_data(data_in::D,sess_set) where D <: RatData
    data = deepcopy(data_in)
    @unpack new_sess,new_sess_free,sess_inds,sess_inds_free = data
    sess_full = use_sessions(sess_set,new_sess,sess_inds)
    size_full = size(new_sess)
    sess_free = use_sessions(sess_set,new_sess_free,sess_inds_free)
    size_free = size(new_sess_free)

    fields = fieldnames(D)
    for fld in fields
        field = getfield(data,fld)
        if typeof(field) <: AbstractArray
            if size(field) == size_full
                keepat!(field,sess_full)
            elseif size(field) == size_free
                keepat!(field,sess_free)
            end
        end
    end

    data = Parameters.reconstruct(data,ntrials=length(new_sess),nfree=length(new_sess_free),sess_inds=get_sess_inds(new_sess),sess_inds_free=get_sess_inds(new_sess_free))
    return data
end

"""
    split_data(data::RatData,train_set::Vector{Int},test_set::Vector{Int})

Split RatData struct `data` into training and test sets specified by `train_set` and `test_set`
"""
function split_data(data::RatData,train_set::Vector{Int},test_set::Vector{Int})
    data_train = trim_data(data,train_set)
    data_test = trim_data(data,test_set)
    return data_train,data_test
end

"""
    nfold_sessions(data::RatData; method::String="stride",nfold::Int=3,seed::Int=0)

Split RatData struct `data` into `nfold` sets using method specified by `method`. Returns train and test set `n` (default = 1)
"""
function split_data(data::RatData,n::Int=1,type="leave_out";kwargs...)
    if type=="leave_out"
        train_set,test_set = leave_out_sessions(data;kwargs...)
    elseif type=="nfold"
        train_set,test_set = nfold_sessions(data;kwargs...)
    else
        error("unknown splitting method")
    end
    data_train,data_test = split_data(data,train_set[n],test_set[n])
    return data_train,data_test
end

function nfold_sessions(data::RatData; method="stride",nfold=3,seed=0)
    # split out train/test sets into `nfold` sets
    @unpack new_sess = data

    nsess = sum(new_sess)
    nset = Int(round(nsess/nfold))

    # split out train/test sets
    sess_use = collect(1:nsess)
    test_set = Array{Vector{Int}}(undef,nfold)
    train_set = Array{Vector{Int}}(undef,nfold)
    if method=="random"
        if seed > 0
            Random.seed!(seed)
        end
        shuffle!(sess_use)
        start = 1
        for n = 1:nfold-1
            test_set[n] = sort(sess_use[start:start+nset-1])
            train_set[n] = sort(setdiff(sess_use,test_set[n]))
            start += nset
        end
        test_set[nfold] = sort(sess_use[start:end])
        train_set[nfold] = sort(setdiff(sess_use,test_set[nfold]))
    elseif method=="stride"
        for n = 1:nfold
            test_set[n] = sess_use[n:nfold:end]
            train_set[n] = setdiff(sess_use,test_set[n])
        end
    else
        error("unknown splitting method")
    end

    return train_set,test_set
end

function leave_out_sessions(data::RatData; method="stride",nout=3,seed=0)
    # split out train/test sets by removing `nout` sessions from each set.
    @unpack new_sess = data

    nsess = sum(new_sess)
    nset = Int(ceil(nsess/nout))
    step = Int(floor(nsess/nout))

    # split out train/test sets
    sess_use = collect(1:nsess)
    test_set = Array{Vector{Int}}(undef,nset)
    train_set = Array{Vector{Int}}(undef,nset)
    if method=="random"
        if seed > 0
            Random.seed!(seed)
        end
        shuffle!(sess_use)
        start = 1
        for n = 1:nset
            test_set[n] = sess_use[start:start+nout-1]
            train_set[n] = sort(setdiff(sess_use,test_set[n]))
            start += nout
        end
    elseif method=="stride"
        test_set[1] = sess_use[1:nset:nsess]
        train_set[1] = setdiff(sess_use,test_set[1])
        for n = 2:nset
            test_set[n] = sess_use[n:step:nsess]
            train_set[n] = setdiff(sess_use,test_set[n])
        end
    else
        error("unknown splitting method")
    end

    return train_set,test_set
end
