"""
    vectorize(vars_in,scale)

Concatenates tuple of parameters into single vector. Use optional input `scale` to indicate
whether to put parameter through a transformation function before concatenation. Scale = 1 for logit, 2 for log2.
"""
function vectorize(vars_in::AbstractVector,scale::AbstractVector) #where T <: Union{Float64,ForwardDiff.Dual}
    vars = deepcopy(vars_in)
    for i in eachindex(scale)
        if any(scale[i] .> 0)
            vars[i] .= vrescale.(vars[i],scale[i])
        end
    end
    return mapreduce(v->vec(v),vcat,vars)
end

function vrescale(var::V,scale::Int64) where V <: Real
    # rescale function for vectorize()
    if scale == 1
        var = logit(var)
    elseif scale == 2
        var = log(2,var)
    end 
    return var
end

"""
    extract(v_in,vars,scale)

Extract vectorized parameters accoring to shapes pulled from `vars`.
Use `scale` to indicate whether parameter was passed through a transformation function. Scale = 1 for logit, 2 for log2.
"""
function extract(v_in::AbstractVector,vars::AbstractVector,scale::AbstractVector) 
    v = copy(v_in)
    return (reshape!(v,var,s) for (var,s) in zip(vars,scale))
end

function reshape!(v::AbstractVector,var::AbstractArray,scale::Union{Int64,Array{Int64}}=0) 
    # reshape function for extract()
    if any(scale .> 0)
        # varout = erescale(reshape(v[1:length(var)],size(var)),scale)
        varout = erescale.(SizedVector{size(var)...}(v[1:length(var)]),scale)

    else
        # varout = reshape(v[1:length(var)],size(var))
        varout = SizedMatrix{size(var)...}(v[1:length(var)])
    end
    deleteat!(v,1:length(var))
    return varout
end

function erescale(var::V,scale::Int64) where V <: Union{Float64,ForwardDiff.Dual}
    # rescale function for extract()
    if scale == 1
        var = logistic(var)
    elseif scale == 2
        var = 2 ^ (var)
    end
    return var
end
 
"""
    βT_x(β,x)

Calculates `transpose(β) * x` according to shape of `β` and `x`
"""
function βT_x(β::AbstractArray{T1},x::Array{T2}) where {T1 <: Union{Float64,ForwardDiff.Dual}, T2 <: Union{Float64,ForwardDiff.Dual}}
    # if size(β)==size(x)
    #     return sum(β .* x, dims=1)
    if (size(x,1)==length(β)) & (size(β,2) > 1) # reshape x to 3D matrix if there's as many rows as the length of β
        return βT_x(β,reshape(x,(size(β)...,:)))
    else
        return β' * x
    end
end

"""
    βT_x(β,x)

Calculates `transpose(β) * x` when `x` is a 3-dimensional matrix
"""
function βT_x(β::AbstractArray{T1},x::Array{T2,3}) where {T1 <: Union{Float64,ForwardDiff.Dual}, T2 <: Union{Float64,ForwardDiff.Dual}}
    nstates = size(β,2)
    βx = zeros(nstates,size(x,3))
    for z=1:nstates
        βx += βT_x(β,x[:,z,:])
    end
    return βx
end

"""
    smooth(x::AbstractVector{T},win::Int=5,side::String="left") where T

Function to compute a running average of a vector. `win` is the window around each element to average over. `side` indicates the location of the window: either to the left, middle, or right side of each element.
"""
function smooth(x::AbstractVector{T},win::Int=5,side::String="left") where T
    if side == "left"
        return map(i -> mean(x[max(1,i-win):i]),1:length(x))
    elseif side == "mid"
        return map(i -> mean(x[max(1,i-floor(win/2)):min(length(x),i+ceil(win/2))]),1:length(x))

    elseif side == "right"
        return map(i -> mean(x[i:min(length(x),i+win)]),1:length(x))
    end
end

"""
    smthstd(x,win=5,side="left")

Function to compute a running standard deviation of a vector. `win` is the window around each element to average over. `side` indicates the location of the window: either to the left, middle, or right side of each element.
"""
function smthstd(x::AbstractVector{T},win::Int=5,side::String="left") where T
    if side == "left"
        xsmth = map(i -> std(x[max(1,i-win):i]),1:length(x))
        xsmth[1] = 0
        return xsmth
    elseif side == "mid"
        return map(i -> std(x[max(1,i-floor(win/2)):min(length(x),i+ceil(win/2))]),1:length(x))

    elseif side == "right"
        return map(i -> std(x[i:min(length(x),i+win)]),1:length(x))
    end
end

"""
    smooth(x::AbstractMatrix{T},win::Int=5,dims::Int=1,side::String="left") where T

Same as `smooth` but for matrices. `dims` indicates the dimension to smooth over.
"""
function smooth(x::AbstractMatrix{T},win::Int=5,dims::Int=1,side::String="left") where T
    if dims < 0
        dims = ndims(x)
    end
    return mapslices(i -> smooth(i,win,side),x,dims=dims)
end

"""
    smthstd(x::AbstractMatrix{T},win::Int=5,dims::Int=1,side::String="left") where T

Same as `smthstd` but for matrices. `dims` indicates the dimension to smooth over.
"""
function smthstd(x::AbstractMatrix{T},win::Int=5,dims::Int=1,side::String="left") where T
    if dims < 0
        dims = ndims(x)
    end
    return mapslices(i -> smthstd(i,win,side),x,dims=dims)
end

"""
    onehot(x::AbstractVector{T}) where T

Converts a vector of labels into a one-hot matrix.
"""
function onehot(x::AbstractVector{T}) where T
    return sort(unique(x)) .== permutedims(x)
end

"""
    deleterows(x::AbstractMatrix{T1},v::T2) where {T1, T2 <: Union{AbstractVector{Integer},AbstractVector{Bool}}}

Deletes rows of matrix `x` according to indices in `v`
"""
function deleterows(x::AbstractMatrix{T1},v::T2) where {T1, T2 <: Union{AbstractVector{Integer},AbstractVector{Bool}}}
    if T2 <: AbstractVector{Bool}
        v = findall(v)
    end
    for i in reverse(sort(v))
        x = deleterow(x,i)
    end
    return x
end

"""
    deleterow(x::AbstractMatrix{T1},i::Int) where T1

Deletes row `i` of matrix `x`
"""
function deleterow(x::AbstractMatrix{T1},i::Int) where T1
    return x[1:size(x,1) .!= i,:]
end

"""
    deletecols(x::AbstractMatrix{T1},v::T2) where {T1, T2 <: Union{AbstractVector{Integer},AbstractVector{Bool}}}

Deletes columns of matrix `x` according to indices in `v`
"""
function deletecols(x::AbstractMatrix{T1},v::T2) where {T1, T2 <: Union{AbstractVector{Integer},AbstractVector{Bool}}}
    if T2 <: AbstractVector{Bool}
        v = findall(v)
    end
    for i in reverse(sort(v))
        x = deletecol(x,i)
    end
    return x
end

"""
    deletecol(x::AbstractMatrix{T1},i::Int) where T1

Deletes column `i` of matrix `x`
"""
function deletecol(x::AbstractMatrix{T1},i::Int) where T1
    if T2 <: AbstractVector{Bool}
        i = findall(i)
    end
    return x[:,1:size(x,2) .!= i]
end

################################################################
########--     roughly translated from psytrack     --##########
################################################################

function D_v(v)
    Dv = copy(hcat(v[:,1],diff(v,dims=2)))
    #Dv[:,new_sess] .= copy(v[:,new_sess])
    return vec(Dv)
end

function DT_v(v,K)
    v = reshape(v, (K,:))
    return DT_v(v)
end

function DT_v(v)
    vf = reverse(v,dims=2)
    DTv = reverse(hcat(vf[:,1],diff(vf,dims=2)),dims=2)
    #DTv[:,vcat(new_sess[2:end],false)] .= v[:,vcat(new_sess[2:end],false)]
    return vec(DTv)
end

function DT_Σ_D(σ,K)
    diags = copy(reshape(σ,(K,:)))
    main_diag = copy(diags)
    main_diag[:,1:end-1] .+= main_diag[:,2:end]
    main_diag = vec(main_diag)

    off_diags = copy(diags)
    # off_diags[:,1] .= 0 we don't need to pad with zeros because there is no inter-feature interaction in the Kth off diagonal
    # the default concatenation method means the same feature is every K entries
    # so interations between the same feature is on the Kth off diagonal
    # this is different than the python implementation
    off_diags = -vec(off_diags)[K+1:end]
    return spdiagm(-K => off_diags, 0 => main_diag, K => off_diags)
    #return spdiagm(-1 => off_diags, 0 => main_diag, 1 => off_diags)
end

function fill_inverse_σ(σ,σInit,σSess,N,new_sess)
    σinv = repeat(1 ./ (σ .^ 2), 1, N)
    σinv[:,new_sess] = repeat(1 ./ (σSess .^ 2), 1, sum(new_sess))
    σinv[:,1] = 1 ./ (σInit .^ 2)
    return vec(σinv)
    #return mapreduce(a->1 ./ fill(a,N), vcat, σ)
end

# function sparse_log_hess(y,x,N,K)
#     D = spzeros(typeof(x[1]),N*K,N*K)
#     for i=1:N
#         i1 = (i-1)*K + 1
#         i2 = i1 + K - 1
#         D[i1:i2,i1:i2] .= y[i] .* x[:,i] * permutedims(x[:,i])
#     end
#     return D
# end

function sparse_log_hess(y,x,N,K)
    diag = vec(permutedims(y) .* x .* x)
    D = spdiagm(0 => diag)
    for i=2:K
        diag = vcat(spzeros(i-1,N),permutedims(y) .* x[1:end-(i-1),:] .* x[i:end,:])[i:end]
        D .+= spdiagm((i-1) => diag, -(i-1) => diag)
    end
    return D
end

function sparse_logdet(hess)
    lus = lu(hess)
    return sum(log.(abs.(diag(lus.U))) + log.(abs.(diag(lus.L))) - log.(abs.(lus.Rs)))
end
# function sparse_log_hess(y,x,N,K)
#     D = spzeros(N*K,N*K)
#     for i=1:K
#         for j=i:K
#             b = spdiagm(y .* x[i,:] .* x[j,:])
#             D[(i-1)*N+1:i*N,(j-1)*N+1:j*N] = copy(b)
#             if i != j
#                 D[(j-1)*N+1:j*N,(i-1)*N+1:i*N] = copy(b)
#             end
#         end
#     end
#     return D
# end

