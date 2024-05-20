using MixtureAgentsModels
include("dirty_plotting.jl")
using GLM,StatsBase,MixedModels,Statistics,Random,Distributions,Interpolations
using HypothesisTests,NaNStatistics
# using mixture_agents_model: initialize_data_full
using Interpolations

function bootci(x)
    med_bs = bootstrap(median,x,BasicSampling(10000))
    med_ci = confint(med_bs,BCaConfInt(0.95))
    return (med_ci[1][1]-med_ci[1][2],med_ci[1][3]-med_ci[1][1])
end

function behavior_forms()
    time_forms = [@formula(d ~ 1 + t + t^2 + t^3), 
        @formula(d ~ 1 + t + t^2 + t^3 + (1 + t + t^2 + t^3)*side_stay),
        @formula(d ~ 1 + t + t^2 + t^2 + (1 + t + t^2 + t^3)*trans), 
        @formula(d ~ 1 + t + t^2 + t^3 + (1 + t + t^2 + t^3)*rew),
        @formula(iti_dur ~ 1 + t + t^2 + t^3 + (1 + t + t^2 + t^3)*rew + ((1 + t + t^2 + t^3 + (1 + t + t^2 + t^3)*rew)|sess))]
    trial_forms = [@formula(d ~ 1 + tr + tr^2 + tr^3), 
        @formula(d ~ 1 + tr + tr^2 + tr^3 + (1 + tr + tr^2 + tr^3)*side_stay),
        @formula(d ~ 1 + tr + tr^2 + tr^3 + (1 + tr + tr^2 + tr^3)*trans), 
        @formula(d ~ 1 + tr + tr^2 + tr^3 + (1 + tr + tr^2 + tr^3)*rew),
        @formula(iti_dur ~ 1 + tr + tr^2 + tr^3 + (1 + tr + tr^2 + tr^3)*rew + ((1 + tr + tr^2 + tr^3 + (1 + tr + tr^2 + tr^3)*rew)|sess))]
    state_forms = [@formula(d ~ z1 + z2), 
        @formula(d ~ 1 + z1 + z2 + (1 + z1 + z2)*side_stay), 
        @formula(d ~ 1 + z1 + z2 + (1 + z1 + z2)*trans), 
        @formula(d ~ 1 + z1 + z2 + (1 + z1 + z2)*rew),
        @formula(iti_dur ~ 1 + z1 + z2 + (1 + z1 + z2)*rew + ((1 + z1 + z2 + (1 + z1 + z2)*rew)|sess))]
    state_time_forms = [@formula(d ~ 1 + t + t^2 + t^3 + z1 + z2), 
        # @formula(d ~ t + t^2 + t^3 + z2 + z3 + z1z2 + z1z3 + z2z3 + (t + t^2 + t^3 + z2 + z3 + z1z2 + z1z3 + z2z3)*cws),
        @formula(d ~ 1 + t + t^2 + t^3 + z1 + z2 + (1 + t + t^2 + t^3 + z1 + z2)*side_stay),
        @formula(d ~ 1 + t + t^2 + t^3 + z1 + z2 + (1 + t + t^2 + t^3 + z1 + z2)*trans),
        @formula(d ~ 1 + t + t^2 + t^3 + z1 + z2 + (1 + t + t^2 + t^3 + z1 + z2)*rew),
        @formula(iti_dur ~ 1 + t + t^2 + t^3 + z1 + z2 + (1 + t + t^2 + t^3 + z1 + z2)*rew + ((1 + t + t^2 + t^3 + z1 + z2 + (1 + t + t^2 + t^3 + z1 + z2)*rew)|sess))]
    state_trial_forms = [@formula(d ~ 1 + tr + tr^2 + tr^3 + z1 + z2), 
        # @formula(d ~ t + t^2 + t^3 + z2 + z3 + z1z2 + z1z3 + z2z3 + (t + t^2 + t^3 + z2 + z3 + z1z2 + z1z3 + z2z3)*cws),
        @formula(d ~ 1 + tr + tr^2 + tr^3 + z1 + z2 + (1 + tr + tr^2 + tr^3 + z1 + z2)*side_stay),
        @formula(d ~ 1 + tr + tr^2 + tr^3 + z1 + z2 + (1 + tr + tr^2 + tr^3 + z1 + z2)*trans),
        @formula(d ~ 1 + tr + tr^2 + tr^3 + z1 + z2 + (1 + tr + tr^2 + tr^3 + z1 + z2)*rew),
        @formula(iti_dur ~ 1 + tr + tr^2 + tr^3 + z1 + z2 + (1 + tr + tr^2 + tr^3 + z1 + z2)*rew + ((1 + tr + tr^2 + tr^3 + z1 + z2 + (1 + tr + tr^2 + tr^3 + z1 + z2)*rew)|sess))]
    no_split_t_forms = [@formula(d ~ 1 + t + t^2 + t^3 + z1 + z2), 
        @formula(d ~ 1 + t + t^2 + t^3 + z1 + z2), 
        @formula(d ~ 1 + t + t^2 + t^3 + z1 + z2), 
        @formula(d ~ 1 + t + t^2 + t^3 + z1 + z2), 
        @formula(iti_dur ~ 1 + t + t^2 + t^3 + z1 + z2 + ((1 + t + t^2 + t^3 + z1 + z2)|sess))]
    no_split_tr_forms = [@formula(d ~ 1 + tr + tr^2 + tr^3 + z1 + z2), 
        @formula(d ~ 1 + tr + tr^2 + tr^3 + z1 + z2), 
        @formula(d ~ 1 + tr + tr^2 + tr^3 + z1 + z2), 
        @formula(d ~ 1 + tr + tr^2 + tr^3 + z1 + z2), 
        @formula(iti_dur ~ 1 + tr + tr^2 + tr^3 + z1 + z2 + ((1 + tr + tr^2 + tr^3 + z1 + z2)|sess))]
    pstate_forms = [@formula(d ~ pz1 + pz2), 
        @formula(d ~ 1 + pz1 + pz2 + (1 + pz1 + pz2)*side_stay), 
        @formula(d ~ 1 + pz1 + pz2 + (1 + pz1 + pz2)*trans), 
        @formula(d ~ 1 + pz1 + pz2 + (1 + pz1 + pz2)*rew),
        @formula(iti_dur ~ 1 + pz1 + pz2 + (1 + pz1 + pz2)*rew + ((1 + pz1 + pz2 + (1 + pz1 + pz2)*rew)|sess))]
    pstate_time_forms = [@formula(d ~ 1 + t + t^2 + t^3 + pz1 + pz2), 
        @formula(d ~ 1 + t + t^2 + t^3 + pz1 + pz2 + (1 + t + t^2 + t^3 + pz1 + pz2)*side_stay),
        @formula(d ~ 1 + t + t^2 + t^3 + pz1 + pz2 + (1 + t + t^2 + t^3 + pz1 + pz2)*trans),
        @formula(d ~ 1 + t + t^2 + t^3 + pz1 + pz2 + (1 + t + t^2 + t^3 + pz1 + pz2)*rew),
        @formula(iti_dur ~ 1 + t + t^2 + t^3 + pz1 + pz2 + (1 + t + t^2 + t^3 + pz1 + pz2)*rew + ((1 + t + t^2 + t^3 + pz1 + pz2 + (1 + t + t^2 + t^3 + pz1 + pz2)*rew)|sess))]

    forms = [time_forms,trial_forms,state_forms,state_time_forms,state_trial_forms,no_split_t_forms,no_split_tr_forms,pstate_forms,pstate_time_forms]
    form_labels = ["time","trial","state","state_time","state_trial","no_split_t","no_split_tr","pstate","pstate_time"]
    forms = [time_forms,state_forms,state_time_forms,no_split_t_forms,pstate_forms,pstate_time_forms]
    form_labels = ["time","state","state_time","no_split_t","pstate","pstate_time"]

    return forms,form_labels
end

function get_behavior_df(rats)
    fit_i = 2
    fit_fldr = "model_fits_prior_mean"
    sort_args = Dict(:method=>"agent",:a=>1)
    sort_states = true  
    smth_trials = 0
    # time_type = :trial_time

    forms,form_labels = behavior_forms()

    ITI_dur = []
    ITI_var = []
    choice_dur = []
    outcome_dur = []
    rews_all = []
    last_rews_all = []
    trial_times = []
    trial_times_norm = []
    z_all = [[],[],[]]
    pz_all = [[],[],[]]
    z_ind_all = []

    trial_inds = []
    trial_inds_norm = []
    rat_inds_trial = []
    sess_inds_trial = []

    # time_cor = [[],[],[],[]]
    # state_cor = [[],[],[],[]]
    # state_time_cor = [[],[],[],[]]
    cors_all = [[],[],[],[],[],[],[],[],[]]
    cors_label = [[],[],[],[],[],[],[],[],[]]
    sse_all = [[],[],[],[],[],[],[],[],[]]
    r2_all = [[],[],[],[],[],[],[],[],[]]
    form_i = [2,3,4,1,1,1,1,1,1]

    sess_inds = []
    rat_inds = []
    trans_all = []
    last_trans_all = []
    last_free_all = []
    stay_all = []
    side_stay_all = []
    mb_stay_all = []
    mb_wsls_all = []
    cws_all = [] # common-win-stay
    cwsls_all = [] # common-win-stay-lose-switch

    for rat_i in rats
        println(rat_i)
        model,agents,model_ops,agent_ops,data = load_model_fit(rat_i,fit_i;data_fldr=fit_fldr,sort_states=sort_states,sort_args...)
    
        γs,_,_ = compute_posteriors(model,agents,data)
        z_ind = argmax.(eachcol(γs))
        zs = onehot(z_ind)

        ikeep = []
        for (sess_i,(inds,ifree)) in enumerate(zip(data.sess_inds,data.sess_inds_free))
            inds_c = copy(inds[2:end-1])
            free = (data.forced[inds_c] .!= 1) #.& (ctimes .< 10)
            if data.forced[inds[1]] && data.forced[inds[end]]
                i = copy(ifree)
                ikeep = vcat(ikeep,i)
            elseif data.forced[inds[1]] && (data.forced[inds[end]] .== 0)
                i = copy(ifree[1:end-1])
                ikeep = vcat(ikeep,i)
            elseif (data.forced[inds[1]] .== 0) && data.forced[inds[end]]
                i = copy(ifree[2:end])
                ikeep = vcat(ikeep,i)
            else
                i = copy(ifree[2:end-1])
                ikeep = vcat(ikeep,i)
            end
            starts = data.step1_times[inds_c]
            ends = data.outcome_times[inds_c]
            rews = data.rewards[inds_c][free] .== 1
            trans = data.trans_commons[inds_c][free]
            last_rews = data.rewards[inds_c .- 1][free] .== 1
            last_trans = data.trans_commons[inds_c .- 1][free]
            last_free = (data.forced[inds_c .- 1] .!= 1)[free] #.& (ctimes .< 10)
            stay = (data.choices[inds_c .- 1] .== data.choices[inds_c])[free]
            side_stay = (data.outcomes[inds_c .- 1] .== data.choices[inds_c])[free]

            mb_stay = ((stay .== 1) .& (last_trans .== 1)) .| ((stay .== 0) .& (last_trans .== 0))
            mb_wsls = ((mb_stay .== 1) .& (last_rews .== 1)) .| ((mb_stay .== 0) .& (last_rews .== 0))
            cws = (stay .== 1) .& (last_trans .== 1) .& (last_rews .== 1)
            cwsls = ((stay .== 1) .& (last_trans .== 1) .& (last_rews .== 1)) .| ((stay .== 0) .& (last_trans .== 1) .& (last_rews .== 0))
            mb_stay_all = vcat(mb_stay_all,mb_stay)
            mb_wsls_all = vcat(mb_wsls_all,mb_wsls)
            cws_all = vcat(cws_all,cws)
            cwsls_all = vcat(cwsls_all,cwsls)
            last_rews_all = vcat(last_rews_all,last_rews)
            last_trans_all = vcat(last_trans_all,last_trans)
            last_free_all = vcat(last_free_all,last_free)
            stay_all = vcat(stay_all,stay)
            side_stay_all = vcat(side_stay_all,side_stay)
            ctimes = (data.choice_times .- data.step1_times)[inds_c][free]
            otimes = (data.outcome_times .- data.step2_times)[inds_c][free]

            rews_all = vcat(rews_all,rews)
            trans_all = vcat(trans_all,trans)

            ttimes = (starts .- starts[1])[free]
            trial_times = vcat(trial_times,ttimes ./ 60)

            ttimes_norm = ttimes ./ starts[end]
            trial_times_norm = vcat(trial_times_norm,ttimes_norm)

            c_dur = log10.(ctimes)
            choice_dur = vcat(choice_dur,c_dur)

            o_dur = log10.(otimes)
            outcome_dur = vcat(outcome_dur,o_dur)
    

            sf = data.step1_times[inds_c .+ 1]
            ef = data.outcome_times[inds_c]
            rf = copy(rews)
            ITI_d = ((sf .- ef) .^ (-1))[free]
            # ITI_d = log10.(sf .- ef)[free]


            ITI_v = zeros(length(ITI_d))

            i_rew = findall(rf .== 1) 
            ITI_v[i_rew[2:end]] = (ITI_d[i_rew[2:end]] .- ITI_d[i_rew[1:end-1]]) .^2 #./ ITI_d[i_rew[1:end-1]]
            i_om = findall(rf .== 0) 
            ITI_v[i_om[2:end]] = (ITI_d[i_om[2:end]] .- ITI_d[i_om[1:end-1]]) .^2 #./ ITI_d[i_om[1:end-1]] 
            # ITI_d = ITI_d[2:end][free]
            # ITI_v = ITI_v[2:end][free]
            
            ITI_dur = vcat(ITI_dur,ITI_d)
            ITI_var = vcat(ITI_var,ITI_v)

            trials = inds_c[free] .- (inds[1] - 1)
            trial_inds = vcat(trial_inds,trials)

            trials_norm = trials ./ (inds[end] - (inds[1] - 1))
            trial_inds_norm = vcat(trial_inds_norm,trials_norm)

            sess_inds_trial = vcat(sess_inds_trial,sess_i .* ones(sum(free)))
            rat_inds_trial = vcat(rat_inds_trial,rat_i .* ones(sum(free)))

            for (d,(dur,idx)) in enumerate(zip([c_dur,o_dur,ITI_d,c_dur[side_stay .== 0],c_dur[side_stay .== 1],o_dur[trans .== 0],o_dur[trans .== 1],ITI_d[rews .== 0], ITI_d[rews .== 1]],[1:length(c_dur),1:length(o_dur),1:length(ITI_d),side_stay .== 0, side_stay .== 1, trans .== 0, trans .== 1,rews .== 0, rews .== 1]))
                df_line = DataFrame(d=Float64.(dur[:]),
                    t=Float64.(ttimes_norm[idx][:]),
                    z1=zs[1,i][idx],
                    pz1=γs[1,i][idx],
                    z2=zs[2,i][idx],
                    pz2=γs[2,i][idx],
                    z3=zs[3,i][idx],
                    pz3=γs[3,i][idx],
                    tr=Float64.(trials_norm[idx][:]),
                    trans=trans[idx],
                    rew=rews[idx],
                    side_stay=side_stay[idx])

                for (form_set,form_label) in zip(forms,form_labels)
                    # println(form_set[form_i[d]])
                    lmm = lm(form_set[form_i[d]],df_line)
                    dur_line = predict(lmm)
                    dur_cor = cor(dur_line,dur)

                    dur_sse = deviance(lmm)
                    dur_r2 = r2(lmm)

                    cors_all[d] = vcat(cors_all[d],dur_cor)
                    sse_all[d] = vcat(sse_all[d],dur_sse)
                    r2_all[d] = vcat(r2_all[d],dur_r2)
                    cors_label[d] = vcat(cors_label[d],form_label)
                end
            end
            sess_inds = vcat(sess_inds,ones(length(form_labels)) .* sess_i)
            rat_inds = vcat(rat_inds,ones(length(form_labels)) .* rat_i)
        end


        for zi = 1:3
            z_all[zi] = vcat(z_all[zi],zs[zi,ikeep])
            pz_all[zi] = vcat(pz_all[zi],γs[zi,ikeep])
        end
        z_ind_all = vcat(z_ind_all,z_ind[ikeep])
    end


    df_trials = DataFrame(
        rat=string.(Int.(rat_inds_trial[:])),
        sess=string.(Int.(sess_inds_trial[:])),
        trial=Int.(trial_inds[:]),
        tr=Float64.(trial_inds_norm[:]),
        trial_time=Float64.(trial_times[:]),
        t=Float64.(trial_times_norm[:]),
        z1=Float64.(z_all[1][:]),
        z2=Float64.(z_all[2][:]),
        z3=Float64.(z_all[3][:]),
        pz1=Float64.(pz_all[1][:]),
        pz2=Float64.(pz_all[2][:]),
        pz3=Float64.(pz_all[3][:]),
        t_choice=Float64.(choice_dur[:]),
        t_outcome=Float64.(outcome_dur[:]),
        iti_dur=Float64.(ITI_dur[:]),
        iti_var=Float64.(ITI_var[:]),
        rew=Int.(rews_all[:]),
        trans=Int.(trans_all[:]),
        last_rew = Int.(last_rews_all[:]),
        last_trans = Int.(last_trans_all[:]),
        last_free = Int.(last_free_all[:]),
        # last_pred_err = Float64.(last_pred_err_all[:]),
        stay = Int.(stay_all[:]),
        side_stay = Int.(side_stay_all[:]),
        mb_stay = Int.(mb_stay_all[:]),
        mb_wsls = Int.(mb_wsls_all[:]),
        cws = Int.(cws_all[:]),
        cwsls = Int.(cwsls_all[:]),
        z_ind = Int.(z_ind_all[:])
    )

    df_cor = DataFrame(
        rat = string.(Int.(rat_inds[:])),
        sess = string.(Int.(sess_inds[:])),
        cor_label = string.(cors_label[1][:]),

        choice_cor = Float64.(cors_all[1][:]),
        choice_cor_w = Float64.(cors_all[4][:]),
        choice_cor_s = Float64.(cors_all[5][:]),

        choice_sse = Float64.(sse_all[1][:]),
        choice_sse_w = Float64.(sse_all[4][:]),
        choice_sse_s = Float64.(sse_all[5][:]),

        choice_r2 = Float64.(r2_all[1][:]),
        choice_r2_w = Float64.(r2_all[4][:]),
        choice_r2_s = Float64.(r2_all[5][:]),

        outcome_cor = Float64.(cors_all[2][:]),
        outcome_cor_u = Float64.(cors_all[6][:]),
        outcome_cor_c = Float64.(cors_all[7][:]),

        outcome_sse = Float64.(sse_all[2][:]),
        outcome_sse_u = Float64.(sse_all[6][:]),
        outcome_sse_c = Float64.(sse_all[7][:]),

        outcome_r2 = Float64.(r2_all[2][:]),
        outcome_r2_u = Float64.(r2_all[6][:]),
        outcome_r2_c = Float64.(r2_all[7][:]),

        iti_cor = Float64.(cors_all[3][:]),
        iti_cor_o = Float64.(cors_all[8][:]),
        iti_cor_r = Float64.(cors_all[9][:]),

        iti_sse = Float64.(sse_all[3][:]),
        iti_sse_o = Float64.(sse_all[8][:]),
        iti_sse_r = Float64.(sse_all[9][:]),

        iti_r2 = Float64.(r2_all[3][:]),
        iti_r2_o = Float64.(r2_all[8][:]),
        iti_r2_r = Float64.(r2_all[9][:]),
    )
    return df_cor,df_trials
end

df_cor,df_trial = get_behavior_df([7])
df_cors,df_trials = get_behavior_df(vcat(collect(1:19),21))

function get_behavior_devs(df_trials)

    forms,form_labels = behavior_forms()
    form_i = 5
    rat_inds = []
    sse_all = []
    dev_all = []
    varest_all = []
    # r2_all = []
    label_all = []
    lmms_all = []


    for df_trial in groupby(df_trials,:rat)
        println(df_trial.rat[1])

    # df_grp = groupby(df_trials,:rat)

        for (form_set,form_label) in zip(forms,form_labels)
            lmm = fit(MixedModel,form_set[form_i],df_trial)
            dur_dev = deviance(lmm)
            dur_sse = sum((predict(lmm) .- df_trial.iti_dur) .^ 2)
            dur_varest = varest(lmm)

            # dur_r2 = r2(lmm)
            sse_all = vcat(sse_all,dur_sse)
            dev_all = vcat(dev_all,dur_dev)
            varest_all = vcat(varest_all,dur_varest)
            # r2_all = vcat(r2_all,dur_r2)
            label_all = vcat(label_all,form_label)
            rat_inds = vcat(rat_inds,df_trial.rat[1])
            lmms_all = vcat(lmms_all,lmm)
        end
    end

    df_devs = DataFrame(
        rat = string.(rat_inds[:]),
        sse = Float64.(sse_all[:]),
        dev = Float64.(dev_all[:]),
        varest = Float64.(varest_all[:]),
        label = string.(label_all[:]),
        lmm = lmms_all[:]
    )
    return df_devs

end
df_devs = get_behavior_devs(df_trials) # very slow, should think about saving the output

function plot_behavior_cpd!(df_devs,type = "state_time",side=nothing,shift=0,c=[1 2])
    # type = "state_time"
    # for
    if type=="state_time"
        # sse_lab = ["1-time" "2-reward" "3-state"]
        # stats_lab = ["2-time" "3-reward" "1-state"]
        # sse_comp = [("state_time",),("time",),("no_split_t",),("state",)]

        sse_lab = ["1-time" "2-state"]
        stats_lab = ["2-time" "1-state"]
        sse_comp = [("state_time",),("time",),("state",)]
    elseif type=="pstate_time"
        sse_lab = ["1-time" "2-state"]
        stats_lab = ["2-time" "1-state"]
        sse_comp = [("pstate_time",),("time",),("pstate",)]
    elseif type=="pstate_time2"
        sse_lab = ["1-ptime" "2-pstate"]
        stats_lab = ["2-ptime" "1-pstate"]
        sse_comp = [("pstate_time",),("time",),("pstate",)]
    elseif type=="state_trial"
        sse_lab = ["1-trial" "2-reward" "3-state"]
        stats_lab = ["2-trial" "3-reward" "1-state"]
        sse_comp = [("state_trial",),("trial",),("no_split_tr",),("state",)]
    end

    keymap = groupby(df_devs,:label).keymap
    comp_inds = [keymap[ip] for ip in sse_comp]
    df_full = groupby(df_devs,:label)[comp_inds[1]]
    df_comp = groupby(df_devs,:label)[comp_inds[2:end]]

    sse_full = df_full[!,:sse]
    cpd_labs = repeat(sse_lab,size(sse_full,1))
    stats_labs = repeat(stats_lab,size(sse_full,1))
    # mulmed(x) = median(100 .* x)

    cpd = zeros(length(sse_full),length(df_comp))

    for (d,df) in enumerate(df_comp)
        cpd[:,d] .= 100 .* (df[!,:sse] .- sse_full) ./ df[!,:sse]
    end

    rat = repeat(df_full.rat,1,length(comp_inds[2:end]))


    df_cpd = DataFrame(
        rat = rat[:],
        cpd = reverse(cpd,dims=2)[:],
        cpd_label = cpd_labs[:],
        stats_label = stats_labs[:])

    df_avg = combine(groupby(df_cpd,:cpd_label),:cpd=>median,:cpd=>bootci)
    @df df_cpd violin!(:cpd_label,:cpd,group=:cpd_label,c=c,side=side,alpha=0.5,label="",framestyle=:box)
    @df df_cpd dotplot!(:cpd_label,:cpd,group=:cpd_label,side=side,mc=c,msw=0,label="",ms=5)
    @df df_avg scatter!([0.5,1.9] .+ shift,:cpd_median,yerror=:cpd_bootci,color=:black,fc=:gray33,markersize=7,lw=3,label="")
    yticks!(0:5:25,["0%","5%","10%","15%","20%","25%"])
    return df_cpd
end

function plot_state_coefs(df_devs)
    nice_blue = RGB(65/255,105/255,225/255)
    nice_red = RGB(178/255,34/255,34/255)

    nice_blue_d = RGB(5/255,45/255,165/255)
    nice_red_d = RGB(138/255,0/255,0/255)

    zl = repeat([["z1" "z1"]; ["z2" "z2"]],1,1,20)
    zl2 = repeat([["2-z1" "2-z1"]; ["1-z2" "1-z2"]],1,1,20)

    rl = repeat([["om" "rew"]; ["om" "rew"]],1,1,20)
    rl2 = repeat([["2-om" "1-rew"]; ["2-om" "1-rew"]],1,1,20)
    rati = zeros(2,2,20)
    zs = zeros(2,2,20)
    for (i,grp) in enumerate(groupby(df_devs,:rat))
        lmm = groupby(grp,:label)[3].lmm[1] 
        zs[:,:,i] .= [[coef(lmm)[5] coef(lmm)[5]+coef(lmm)[11]]; [coef(lmm)[6] coef(lmm)[6]+coef(lmm)[12]]]
        rati[:,:,i] .= i
    end
    df = DataFrame(
        rat = string.(Int.(rati[:])),
        z = string.(zl[:]),
        z2 = string.(zl2[:]),
        r = string.(rl[:]),
        r2 = string.(rl2[:]),
        coef = Float64.(zs[:])
    )

    df_avg = combine(groupby(df,[:r,:z]),:coef=>median,:coef=>bootci)
    p = @df groupby(df,:r)[1] violin(:z,:coef,group=:z,framestyle=:box,side=:left,msw=0,c=nice_red,alpha=0.5,label="")
    print(groupby(df,:r)[1])
    @df groupby(df,:r)[2] violin!(:z,:coef,group=:z,framestyle=:box,side=:right,msw=0,c=nice_blue,alpha=0.5,label="")
    @df groupby(df,:r)[1] dotplot!(:z,:coef,group=:z,framestyle=:box,side=:left,msw=0,c=nice_red_d,label="")
    @df groupby(df,:r)[2] dotplot!(:z,:coef,group=:z,framestyle=:box,side=:right,msw=0,c=nice_blue_d,label="")
    @df groupby(df_avg,:r)[1] scatter!([0.5,1.5] .- 0.15,:coef_median,yerror=:coef_bootci,color=:black,fc=:gray33,markersize=7,lw=3,label="")
    @df groupby(df_avg,:r)[2] scatter!([0.5,1.5] .+ 0.15,:coef_median,yerror=:coef_bootci,color=:black,fc=:gray33,markersize=7,lw=3,label="")


    hline!([0],c=:black,label="")
    ylims!(-0.16,0.16)

    return p,df
end
plot_state_coefs(df_devs)[1]


function plot_state_rectangles!(p,z,z_1h,inds,height)
    rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
    for (iz,i) in enumerate(sort(unique(z)))
        o = findall(diff(z_1h[iz,:]) .== 1)
        f = findall(diff(z_1h[iz,:]) .== -1)
        if !isempty(f) || !isempty(o)
            if (length(f) > length(o)) 
                o = vcat(1,o)
            end
            if (length(o) > length(f)) 
                f = vcat(f,length(z[:]))
            end
            if (o[1] > f[1]) && (o[end] > f[end])
                o = vcat(1,o)
                f = vcat(f,length(z[:]))
            end
        end

        for r = 1:length(o)
            plot!(p,rectangle(inds[f[r]]-inds[o[r]],height,inds[o[r]],0),alpha=0.2,linewidth=0,c=i,label="")
        end
    end
end

function plot_behavior_regs(df_trial,lmm,ex_sess=108,type="time",cs=[1 2 3])
    forms,_ = behavior_forms()
    nice_blue = RGB(65/255,105/255,225/255)
    nice_red = RGB(178/255,34/255,34/255)

    nice_blue_d = RGB(5/255,45/255,165/255)
    nice_red_d = RGB(138/255,0/255,0/255)

    if type == "time"
        form_set = 4
        t_type = :t
        z_type = [:z1,:z2,:z3]
        time_type = :trial_time
        xlab = "time (min)"
    elseif type == "trial"
        form_set = 5
        t_type = :tr
        z_type = [:z1,:z2,:z3]
        time_type = :trial
        xlab = "trial"
    elseif type == "pstate"
        form_set = 9
        t_type = :t
        z_type = [:pz1,:pz2,:pz3]
        time_type = :trial_time
        xlab = "time (min)"
    end
    vars_trial = [:iti_dur]
    vars_pred = [:iti_pred]
    vars_split = [:rew]
    form_i = [4]
    # for vpred in vars_pred
    df_trial[!,vars_pred[1]] = predict(lmm)
    # end
    df_ex = groupby(df_trial,:sess)[ex_sess]


    p = Array{Any}(undef,length(vars_trial))

    for (ip,(var2,vpred,split)) in enumerate(zip(vars_trial,vars_pred,vars_split))

        dur = df_ex[!,var2]

        df_line = DataFrame(d=dur,t=df_ex.t,tr=df_ex.tr,z1=df_ex.z1,z2=df_ex.z2,z3=df_ex.z3,rew=df_ex.rew,trans=df_ex.trans,side_stay=df_ex.side_stay,z_ind=df_ex.z_ind,pz1=df_ex.pz1,pz2=df_ex.pz2,pz3=df_ex.pz3)

        if !isnothing(split)
            # lmm = lm(forms[form_set][form_i[ip]],df_line)
            # print(lmm)
            i1 = df_ex[!,split] .== 1
            i2 = i1 .== 0

            D = Array(df_line[!,[t_type,z_type...,split]])

            t_pred = (repeat(D[:,1],1,3) .^ [1 2 3]) * coef(lmm)[2:4] 
            t_pred_rew = t_pred .+ (repeat(D[:,1],1,3) .^ [1 2 3]) * coef(lmm)[8:10] 
            s_pred = D[:,2:3] * coef(lmm)[5:6] 
            s_pred_rew = s_pred .+ (D[:,2:3] * coef(lmm)[11:12])            

            p1 = plot()
            plot_state_rectangles!(p1,df_ex.z_ind,onehot(df_ex.z_ind),df_ex[!,time_type],maximum(dur))

            scatter!(df_ex[!,time_type][i1],dur[i1],c=nice_blue,msw=0)#,yflip=true)
            # yticks!(0:1:3,[L"$10^0$",L"$10^1$",L"$10^2$",L"$10^3$"])
            scatter!(df_ex[!,time_type][i2],dur[i2],c=nice_red,msw=0)
            # plot!(df_ex[!,time_type][i1],predict(lmm)[i1],lw=4,c=:black,s=:dash,label="full")
            # plot!(df_ex[!,time_type][i2],predict(lmm)[i2],lw=4,c=:gold4,label="full")
            plot!(df_ex[!,time_type][i1],df_ex[!,vpred][i1],lw=4,c=nice_blue_d,s=:dash,label="full") #s=:dash
            plot!(df_ex[!,time_type][i2],df_ex[!,vpred][i2],lw=4,c=nice_red_d,label="full")

            plot!(xformatter=_->"")
            plot!(legend=false)
            # hline!([coef(lmm)[1]],c=:gold3,label="")
            # hline!([coef(lmm)[1] .+ coef(lmm)[7]],c=:black,s=:dash,label="")


            p2 = plot(df_ex[!,time_type],t_pred,lw=4,c=:black,label="time")#,yflip=true)
            plot!(df_ex[!,time_type],t_pred_rew,lw=4,c=:black,s=:dash,label="time")#,yflip=true)
            plot!(df_ex[!,time_type],s_pred,lw=4,c=cs[3],label="state")#,yflip=true)
            plot!(df_ex[!,time_type],s_pred_rew,lw=4,c=cs[3],s=:dash,label="state")#,yflip=true)
            # hline!([coef(lmm)[1]],c=:gold3,label="")
            # hline!([coef(lmm)[1] .+ coef(lmm)[7]],c=:black,s=:dash,label="")
            plot!(legend=false)
            xlabel!(xlab)
            # yticks!([0,1],[L"$\times 10^0$",L"$\times 10^1$"])
            plot!(legend=false)

            dur_c = dur .- t_pred .- D[:,end] .* ((repeat(D[:,1],1,3) .^ [1 2 3]) * coef(lmm)[8:10])
            df_stats = DataFrame(d=dur_c,rew=df_ex.rew,trans=df_ex.trans,side_stay=df_ex.side_stay,z_ind=df_ex.z_ind,z=string.(df_ex.z_ind),sess=df_ex.sess)
            df_z = groupby(df_stats,:z_ind)
            p3 = plot()
            for df in df_z
                df_s = groupby(df,split)
                df_avg = combine(df_s,:d=>median=>:d_m,:d=>bootci=>:d_e)

                for (s,(side,c,shift)) in enumerate(zip([:left,:right],[:gold3,:gray],[-0.1,0.1]))
                    violin!(df_s[s].z_ind,df_s[s].d,c=c,side=side,alpha=0.5,label="",framestyle=:box)#,yflip=true)
                    dotplot!(df_s[s].z_ind,df_s[s].d,mc=c,side=side,msw=0,label="",ms=5)
                    # if (s == 1 && df_s[s].z_ind[1] == 1) == false
                    scatter!([df_s[s].z_ind[1]+shift],[df_avg[s,:d_m]],yerror=[df_avg[s,:d_e]],color=:black,markersize=7,lw=3,label="")
                    # end
                end
            end

            # l = @layout [[a{0.7h}; b] c{0.4w}]
            # p[ip] = plot(p1,p2,p3,layout=l,framestyle=:box)

            l = @layout [a{0.7h}; b]
            p[ip] = plot(p1,p2,layout=l,framestyle=:box)

            # ylabel!("log time")
        else
            lmm = lm(forms[form_set][1],df_line)
            print(lmm)

            D = Array(df_line[!,[:tr,:z1,:z2]])
            t_pred = (repeat(D[:,1],1,3) .^ [1 2 3]) * coef(lmm)[2:4] 
            s_pred = D[:,2:3] * coef(lmm)[5:6] 


            p[ip] = scatter(df_ex[!,time_type],dur,c=:gray,msw=0)
            # plot!(df_ex[!,time_type],predict(lmm),lw=4,c=1,label="full")
            plot!(df_ex[!,time_type],t_pred .+ coef(lmm)[1],lw=4,c=c[1],label="time")
            plot!(df_ex[!,time_type],s_pred .+ coef(lmm)[1],lw=4,c=c[1],label="state")
            plot!(df_ex[!,time_type],s_pred .+ t_pred .+ coef(lmm)[1],lw=4,c=:black,label="full")
            plot!(legend=false)
            # plot!(df_ex[!,time_type],predict(lmm),lw=4,c=3,s=:dot,label="")
            hline!([coef(lmm)[1],coef(lmm)[1]],c=:black,s=:dash,label="")

            # ylabel!("log time")
            # xlabel!("time")
            title!(string(var2))
        end


    end
    l = @layout grid(length(vars_trial),1)
    # return p
    return p #plot(p...,layout=l,size=(600,1000),framestyle=:box)

end
#108, 93, 104, 110



# form = @formula iti_dur ~ 1 + t + t^2 + t^3 + z1 + z2 + (1 + t + t^2 + t^3 + z1 + z2)*rew + ((1 + t + t^2 + t^3 + z1 + z2 + (1 + t + t^2 + t^3 + z1 + z2)*rew)|sess)
# form = @formula(iti_dur ~ 1 + t + t^2 + t^3 + z1 + z2 + (1 + t + t^2 + t^3 + z1 + z2)*rew + ((1 + t + t^2 + t^3 + z1 + z2 + (1 + t + t^2 + t^3 + z1 + z2)*rew)|sess))

# lmm = fit(MixedModel,form,df_trial)
# df_trial[!,:iti_pred] = predict(lmm)
lmm = groupby(groupby(df_devs,:rat)[7],:label)[3].lmm[1]
#108,104
p1 = plot(plot_behavior_regs(df_trial,lmm,108,"time")...)

p2 = plot()
# plot_behavior_cpd!(df_devs)
df_cpd = plot_behavior_cpd!(df_devs,"state_time")

# median cpds
combine(groupby(df_cpd,:cpd_label),:cpd=>median)

# stats for cpd
form = @formula cpd ~ 1 + cpd_label + ((1 + cpd_label)|rat) # wrt time
lmm = fit(MixedModel,form,df_cpd)
pvalue(SignedRankTest(Array(groupby(df_cpd,:cpd_label)[1][!,:cpd])))
form = @formula cpd ~ 1 + stats_label + ((1 + stats_label)|rat) # wrt state
lmm = fit(MixedModel,form,df_cpd)
pvalue(SignedRankTest(Array(groupby(df_cpd,:cpd_label)[2][!,:cpd])))


p3,df = plot_state_coefs(df_devs)

# medians of coefficients
combine(groupby(df,[:z,:r]),:coef=>median)

# significant difference between state, z wrt z1, z2 wrt z2
form1 = @formula coef ~ 1 + z2 + ((1 + z2)|rat)
lmm = fit(MixedModel,form1,groupby(df,:r)[1])
pvalue(SignedRankTest(Array(groupby(groupby(df,:z)[2],:r)[1][!,:coef]))) #z2 omission
lmm = fit(MixedModel,form1,groupby(df,:r)[2])
pvalue(SignedRankTest(Array(groupby(groupby(df,:z)[2],:r)[2][!,:coef]))) #z2 reward

# significant difference between reward, r wrt omission, r2 wrt reward
form2 = @formula coef ~ 1 + r + ((1 + r)|rat)
lmm = fit(MixedModel,form2,groupby(df,:z)[1]) #z1
lmm = fit(MixedModel,form2,groupby(df,:z)[2]) #z2
pvalue(SignedRankTest(Array(groupby(groupby(df,:z)[2],:r)[2][!,:coef]) .- Array(groupby(groupby(df,:z)[2],:r)[1][!,:coef]))) #z2 reward


# p2 = plot_behavior_cpd(df_cors)
l = @layout [a b{0.24w} c{0.24w}]
plot(p1,p3,p2,layout=l,size=(1000,500))
title!("")
xlabel!("")
ylabel!("")
# plot!(size=(800,800))
plot!(xtickfontsize=12,ytickfontsize=12)
savefig("/Users/sarah/Library/CloudStorage/Dropbox/Princeton/Manuscripts/MoA-HMM/figures/fig6_behavior_state_cors.svg")

l = @layout [a b{0.33w}]
plot(p1,p3,layout=l,size=(900,500))
title!("")
xlabel!("")
ylabel!("")
# plot!(size=(800,800))
plot!(xtickfontsize=12,ytickfontsize=12)
savefig("/Users/sarah/Library/CloudStorage/Dropbox/Princeton/Presentations/FPO/behavior_state_cors.svg")

