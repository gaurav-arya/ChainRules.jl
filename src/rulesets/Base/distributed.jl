#####
##### `pmap`
#####

# Now that there is a backwards rule for zip (albeit only in Zygote),
# it should be fine to deal with only a single collection X
function rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(pmap), f, p::AbstractWorkerPool, X; kwargs...)
    project_X = ProjectTo(X)

    println("type of X: $(typeof(X))")

    darr = dfill([], (nworkers(p) + 1,), vcat(myid(), workers(p))) # Include own proc to handle empty worker pool

    println("here! with ", nworkers(p), " processors")
    println("eltype of X ", eltype(X))

    function forw(x)
        y, back = rrule_via_ad(config, f, x)
        push!(darr[:L][1], back)
        return y, myid(), length(darr[:L][1])
    end

    print("forward map: ")
    @time ys_IDs_indices = pmap(forw, p, X; kwargs...)

    ys = getindex.(ys_IDs_indices, 1) # the primal values
    IDs = getindex.(ys_IDs_indices, 2) # remember which processors handled which elements of X
    indices = getindex.(ys_IDs_indices, 3) # remember the index of the pullback in the array on each processor

    # create a list of positions in X handled by each processor
    unique_IDs = sort(unique(IDs))
    output_sz = size(ys)
    T = eltype(eachindex(ys_IDs_indices))
    positions = [Vector{T}() for _ in 1:length(unique_IDs)]
    for i in eachindex(ys_IDs_indices)
        push!(positions[searchsortedfirst(unique_IDs, IDs[i])], i)
    end

    function pmap_pullback(Ȳ)
        Ȳ = unthunk(Ȳ)
        println("size of Ȳ: $(size(Ȳ))")
        println("axes of Ȳ: $(axes(Ȳ))")
        println("type of Ȳ: $(typeof(Ȳ))")
        #println("Ȳ[10]: $(Ȳ[10])")

        # runs the pullback for each position handled by proc ID in forward pass
        function run_backs(ID, positions)
            println("Positions: $positions")
            Ȳ_batch = Ȳ[positions]
            indices_batch = indices[positions]
            res_batch = remotecall_fetch(() -> 
                asyncmap((ȳ, i) -> darr[:L][1][i](ȳ), Ȳ_batch, indices_batch), # run all the backs in a local asyncmap
                ID) 
            return res_batch
        end

        # combine the results from each proc into res = pmap((back, ȳ) -> back(ȳ), p, backs for each position, Ȳ)
        print("backward map: ")
        @time res_batches = asyncmap(run_backs, unique_IDs, positions)
        res = similar(res_batches[1], output_sz)
        println("size of res: $(size(res))")
        println("indices: $indices")


        for (positions, res_batch) in zip(positions, res_batches)
            res[positions] = res_batch
        end

        println("Type of first(res[1]): ", typeof(first(res[1])))
        S = first(res[1])
        for i in 2:9
            S += first(res[i])
            println("Sum of first $i: $(typeof(S))")
        end

        # extract f̄ and X̄ 
        println("going to extract fbar")
        f̄ = sum(first, res)
        println("worked!")
        X̄ = project_X(map(last, res))
        return (NoTangent(), f̄, NoTangent(), X̄)
    end

    return ys, pmap_pullback
end

