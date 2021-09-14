module WarningDissem

# Write your package code here.
import Random
import LightGraphs, MetaGraphs
import Distributions
import DataFrames
import DataStructures
import CSV
import JLSO
import GLMakie, Colors
import ProgressMeter

export makenet, disseminate, monte_carlo, sensitivity_analysis
export except, subsetresults, avgdeg, coverage
export draw_disseminate, draw_run, draw_layers, draw_monte_carlo, draw_sensitivity_analysis, draw_grid_monte_carlo, draw_row_sensitivity_analysis
export save, load

const LG = LightGraphs;
const MG = MetaGraphs;
const Dist = Distributions;
const DF = DataFrames;
const DS = DataStructures;
const Makie = GLMakie;
const PM = ProgressMeter;

"""
    unzip(a)

Break apart `a` into two collections.

Since Julia still doesn't have an `unzip()` as of 1.6. This function may become unnecessary in the future.
"""
unzip(a) = map(x -> getfield.(a, x), (fieldnames ∘ eltype)(a));

"""
    window(x, len)

Return a view into an array so you can see more than one element at once.[^src]

# Examples
```jldoctest
julia> WarningDissem.window([1, 2, 3, 4], 2)
3-element Vector{SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}}:
 [1, 2]
 [2, 3]
 [3, 4]
```

[^src]: Mostly pulled from [Stack Overflow](https://stackoverflow.com/a/63769989)
"""
window(x, len) = view.(Ref(x), (:).(1:length(x) - (len - 1), len:length(x)));

"""
    initnet!(G[, layertrusts])

Initialize a network `G` with node properties such as trust.

All properties are only set to the first layer of the network.
"""
function initnet!(G, layertrusts = [.43, .39, .48])
    # All vertex properties are set in the first layer only
    g = G[1];
    for v ∈ LG.vertices(g)
        neighbors = LG.neighbors.(G, v);
        trust = Iterators.flatten(map(enumerate(neighbors)) do (i, neighborset)
            map(neighbor -> ((neighbor, i), layertrusts[i]), neighborset)
        end) |> Dict;
        MG.set_prop!(g, v, :state, NodeState(trust));
    end
end

"""
    makenet(phoneₖ, phoneᵦ, wom_file, womₘₐₓ, smₙ₀, smₖ)

Create a network.
"""
function makenet(phoneₖ, phoneᵦ, wom_file, womₘₐₓ, smₙ₀, smₖ)
    # [Phone, Word of Mouth, Social Media]
    wom, _ = begin
        coords = CSV.File(joinpath("data", wom_file); types = [Float64, Float64]) |> DF.DataFrame;
        coord_matrix = collect(Matrix(coords)');
        LG.euclidean_graph(coord_matrix; cutoff = womₘₐₓ)
    end;
    wom = MG.MetaGraph(wom);

    n = LG.nv(wom);
    phone = LG.watts_strogatz(n, phoneₖ, phoneᵦ) |> MG.MetaGraph;
    sm = LG.barabasi_albert(n, ceil(smₙ₀ * n) |> Int, smₖ) |> MG.MetaGraph;

    [phone, wom, sm]
end

"""
    makenet()

Provide defaults from the literature.
"""
makenet() = makenet(10, .7, "wom_coords.csv", 60., .374, 105);

"""
    sample(x)

Return `x`.

Since many of the parameters to `disseminate()` can take either a distribution or a value, we should
have something that samples the distribution if that's what's provided, or just returns the value.
"""
sample(x) = x;

"""
    sample(x::Distribution)

Return a singular sample of `x`.
"""
sample(x::Dist.Distribution) = Dist.rand(x);

"""
    sample(x, n)

Return an array of length `n` filled with `x`.
"""
sample(x, n) = fill(x, n);

"""
    sample(x::Distribution, n)

Sample `x`, `n` times.
"""
sample(x::Dist.Distribution, n) = Dist.rand(x, n);

@doc raw"""
    shareprob(p)

Provide a probability the individual will share.

Return a function matching the following equation:
```math
p(dₜ, tₗ, c) = c × p₀ × \text{max}\left(1 - \frac{\text{min}(tₗ)}{\text{max}(dₜ, 0)}, 0\right)
```
where max``(x, 0)`` ensures no negative numbers,
min``(x)`` returns the smallest value in ``x``,
``p₀`` is an initial starting probability,
``dₜ`` is the time remaining until the disaster,
``tₗ`` is an array of time it takes to share info on each layer, and
``c`` is the confidence in the info.

The only parameter to this function is ``p₀``. The result of this function is meant to be passed in to
`disseminate()` as the parameter `p`.
"""
function shareprob(p)
    function (dₜ, tₗ, c)
        c * p * max(1 - (minimum ∘ broadcast)(sample, tₗ) / max(dₜ, 0), 0)
    end
end

@doc raw"""
    layerprob(b)

Provide a probability an individual will share on each layer.

Return a function matching the following equation:
```math
pₗ(dₜ, tₗ) = \text{norm}\left(b × \text{max}\left(1 - \frac{tₗ}{\text{max}(dₜ, 0)}, 0\right)\right)
```
where max``(x, 0)`` ensures no negative numbers,
all operations are element-wise across vectors,
norm``(x)`` normalizes the vector ``x`` so its elements sum to 1,
``b`` is an initial starting probability vector whose elements sum to 1,
``dₜ`` is the time remaining until the disaster, and
``tₗ`` is an array of time it takes to share info on each layer.

The only parameter to this function is ``b``. The result of this function is meant to be passed in to
`disseminate()` as the parameter `pₗ`.
"""
function layerprob(b)
    function (dₜ, tₗ)
        unscaled = @. b * max(1 - sample(tₗ) / max(dₜ, 0), 0);
        unscaled ./ sum(unscaled)
    end
end

@doc raw"""
    conflevel(cₙ)

Provide the confidence an individual has on a piece of info.

Return a function matching the following equation:
```math
c(cₛ, wₗ) = \text{clamp}\left(\frac{\text{trust}(cₛ, wₗ)}{cₙ + 1}, 0, 1\right)
```
where clamp``(x, 0, 1)`` enforces results between 0 and 1,
trust``(cₛ, wₗ)`` retrieves the sum of trust weights for the number of times/layers the individual has been informed through,
``cₙ`` is the expected number of times the individual will need to confirm information before sharing it if they completely trust their sources,
``cₛ`` is the number of times/layers the individual has been informed through, and
``wₗ`` is a collection of trust levels for each layer.

The only parameter to this function is `cₙ`. The return value of the returned function ranges between 0 and 1. The returned function is meant to
be passed in to `disseminate()` as the parameter `c`.
"""
function conflevel(cₙ)
    function (cₛ, wₗ)
        # If informed by the broadcast, have the highest confidence level
        if ismissing(cₛ[1][1])
            return 1.
        end
        x = map(i -> wₗ[i], cₛ);
        clamp(sum(x) / (cₙ + 1), 0, 1)
    end
end

@doc raw"""
    evac(tᵣ)

Provide the likelihood an individual will evacuate.

Return a function matching the following equation:
```math
r(dₜ, c) = c × \left(1 - \text{clamp}\left(\frac{dₜ}{tᵣ}, 0, 1\right)\right)
```
where clamp``(x, 0, 1)`` enforces results between 0 and 1,
``tᵣ`` is the maximum time at which anyone would evacuate,
``dₜ`` is the time remaining until the disaster, and
``c`` is the confidence in the info.

The only parameter to this function is `tᵣ`. This parameter is a linear scale to 0, so if 100% confident in the info
then a 50% chance to evac would be ``tᵣ/2`` mins remaining until the disaster and a 100% chance would be 0.

The result of this function is meant to be passed in to `disseminate()` as the parameter `r`.
"""
function evac(tᵣ)
    function (dₜ, c)
        c * (1 - clamp(dₜ / tᵣ, 0, 1))
    end
end

"""
    NodeState(comm::Bool, evac::Bool, informed::Vector, trust::Dict)

The state of each individual.

Every node has an instance of this struct associated with it on the first layer.
"""
mutable struct NodeState
    "If I've started communicating already."
    comm::Bool
    "If I've started evacuating already."
    evac::Bool
    "The collection of individuals who have informed me."
    informed::Vector
    "A dictionary of how much I trust neighboring individuals (keys are tuples of `(node, layer)`)."
    trust::Dict
end

"""
    NodeState(trust::Dict)

The initial state of a node: haven't started communicating, haven't started evacuating, no one has informed me.
"""
NodeState(trust::Dict) = NodeState(false, false, [], trust);

"""
    Comm(t::AbstractFloat, t₀::AbstractFloat, src, srcₗ::Union{Int,Missing})

Properties of a communication between nodes.
"""
struct Comm
    "When the communication ends."
    t::AbstractFloat
    "When the communication starts."
    t₀::AbstractFloat
    "The node that transmits the information."
    src
    "The layer index of the communication (`Missing` if the communication is from the initial broadcast)."
    srcₗ::Union{Int, Missing}
end

# Need these for the priority queue
Base.isless(a::Comm, b::Comm) = isless((a.t, a.t₀), (b.t, b.t₀));
Base.isequal(a::Comm, b::Comm) = isequal((a.t, a.t₀), (b.t, b.t₀));

"""
    hist2cmf(df::DataFrame, col, agg_col)

Convert data in a histogram format to a CMF (cumulative mass function) format.

Given a dataframe (`df`) and a column of that dataframe (`col`), this function counts duplicates and
performs a cumulative sum.

`agg_col` is the name of the new aggregate column.
"""
hist2cmf(df::DF.DataFrame, col, agg_col) = DF.transform(DF.combine(DF.groupby(df, col), DF.nrow => agg_col), agg_col => cumsum; renamecols = false);

"""
    disseminate(G::Vector, n₀::Int, p, pₗ, tₗ, c, r, d; u=Uniform())

Simulate dissemination of emergency warning info through a network `G`.

# Arguments
- `G::Vector`: a vector of `MetaGraph`s where each graph is a layer of the multiplex network. Only the first layer has node properties.
- `n₀::Int`: the initial broadcast size.
- `p`: a function to compute the likelihood of sharing info. (For more details, reference [`shareprob`](@ref).)
- `pₗ`: a function to compute the likelihood of sharing info on each layer. (For more details, reference [`layerprob`](@ref).)
- `tₗ`: a collection of times it will take to share info on each layer. Can be a collection of distributions and/or singular values.
- `c`: a function to compute the confidence of the individual on the new info. (For more details, reference [`conflevel`](@ref).)
- `r`: a function to compute the likelihood of evacuating at this timestep. (For more details, reference [`evac`](@ref).)
- `d`: the total amount of prewarning time before the disaster occurs.
- `u=Uniform()`: the uniform distribution. We can pass it in so the distribution doesn't have to be recreated on every run of the dissemination simulation.
"""
function disseminate(G::Vector, n₀::Int, p, pₗ, tₗ, c, r, d; u::Dist.Distribution = Dist.Uniform())
    # These are the nodes we're starting from
    nodes = Dist.sample(LG.vertices(G[1]), n₀; replace = false);

    events = DS.PriorityQueue();
    DS.enqueue!.(tuple(events), nodes, tuple(Comm(0, 0, missing, missing)));
    informed_times, prob_times, informed_layers, evac_times = [], [], Dict(:t => [], :layer => []), [];

    # Make sure all of the initial properties of the network are set
    initnet!(G);

    while !isempty(events)
        node, comm = DS.dequeue_pair!(events);
        t = comm.t;

        # This is a reference, so any changes will be made inside the graph as well
        state = MG.get_prop(G[1], node, :state);

        push!(state.informed, (comm.src, comm.srcₗ));

        # The node already decided to communicate
        if state.comm
            continue;
        end

        # If this is the first time the node is being informed, log it for the results
        if length(state.informed) == 1
            push!(informed_times, t);
            if !ismissing(comm.srcₗ)
                push!(informed_layers[:t], t);
                push!(informed_layers[:layer], comm.srcₗ);
            end
        end

        conf = c(state.informed, state.trust);

        # If the node decides to evacuate
        if !state.evac && sample(u) ≤ (sample ∘ r)(d - t, conf)
            push!(evac_times, t);

            for neighbor ∈ copy(LG.neighbors(G[2], node))
                # Remove from word-of-mouth only
                success = MG.rem_edge!(G[2], node, neighbor);
                if !success
                    error("Unable to remove node $node from word-of-mouth network (neighbor $neighbor)");
                end
            end

            state.evac = true;
        end

        # If the node decides to communicate to its contacts
        prob = (sample ∘ p)(d - t, tₗ, conf);
        if sample(u) ≤ prob
            push!(prob_times, prob);
            # Pick a layer based on the weights of `pₗ`
            layer = searchsortedfirst((cumsum ∘ broadcast)(sample, pₗ(d - t, tₗ)), sample(u));
            contacts = copy(LG.neighbors(G[layer], node));

            # If reaching out on social media, contact everyone at once
            comm_contacts = if layer == 3
                zip(contacts, fill(sample(tₗ[layer]), length(contacts)))
            else
                Random.shuffle!(contacts);

                # Determine start/end times for communicating with everyone, and zip in contacts
                times = zip(contacts, window(vcat(0, (cumsum ∘ sample)(tₗ[layer], length(contacts))), 2));
                # Once we decide not to inform the next person, we're all done
                # Check with start times, return with end times
                # Checking against 0 because we already decided to communicate at least once (if there are any contacts)
                map(Iterators.takewhile(times) do (_, (tₛ, _))
                        tₛ == 0 || sample(u) ≤ (sample ∘ p)(d - (t + tₛ), tₗ, conf)
                    end) do (cᵢ, (_, tₑ))
                    (cᵢ, tₑ)
                end
            end;

            for (i, tᵢ) in comm_contacts
                # If they're already in the queue, that means they're currently being contacted
                if i ∉ keys(events)
                    DS.enqueue!(events, i, Comm(t + tᵢ, t, node, layer));
                end
            end

            state.comm = true;
        end
    end

    dissem = hist2cmf(DF.DataFrame(t = AbstractFloat.(informed_times)), :t, :dissem);
    probs = hist2cmf(DF.DataFrame(t = AbstractFloat.(prob_times)), :t, :prob);
    evac = hist2cmf(DF.DataFrame(t = AbstractFloat.(evac_times)), :t, :evac);

    # Form layer dissemination data in a more convenient format
    layer_names = [:phone, :wom, :sm];
    temp = DF.DataFrame(informed_layers);
    DF.transform!(temp, :layer => x -> getindex.(tuple(layer_names), x); renamecols = false);
    # Since `unstack` doesn't play nicely with an empty dataframe, do it ourselves
    temp = if isempty(temp)
        DF.select(temp, :t)
    else
        DF.unstack(DF.combine(DF.groupby(temp, [:t, :layer]), DF.nrow), :layer, :nrow);
    end;
    # Prepend the dataframe with a set of zeros
    layers = DF.DataFrame(t = [0], phone = [0], wom = [0], sm = [0]);
    DF.append!(layers, temp; cols = :subset);
    # Replace missing values with 0 since we'll take a cumulative sum on it next
    DF.transform!(layers, x -> coalesce.(x, 0));
    DF.transform!(layers, :phone => cumsum, :wom => cumsum, :sm => cumsum; renamecols = false);

    dissem, probs, layers, evac
end

function monte_carlo(G::Vector, n::Int, n₀::Int, p, pₗ, tₗ, c, r, d; u::Dist.Distribution = Dist.Uniform(), pb = true)
    data = [DF.DataFrame() for _ ∈ 1:4];

    progress_bar = PM.Progress(n; enabled = pb);

    for i ∈ 1:n
        newest = disseminate(deepcopy(G), n₀, p, pₗ, tₗ, c, r, d; u = u);
        DF.insertcols!.(newest, :run => i);
        DF.append!.(data, newest);

        PM.next!(progress_bar);
    end

    data
end

function sensitivity_analysis(n::Int, n₀, p, pₗ, tₗ, c, tᵣ, d; pb = true)
    G = makenet(10, .7, "wom_coords.csv", 60., .374, 105);
    sensitivity_analysis(G, n, n₀, p, pₗ, tₗ, c, tᵣ, d; pb)
end

function sensitivity_analysis(G::Vector, n::Int, n₀, p, pₗ, tₗ, c, tᵣ, d; pb = true)
    data = [DF.DataFrame() for _ ∈ 1:4];
    u = Dist.Uniform();

    progress_bar = PM.Progress(length(n₀) * length(p); enabled = pb);

    for n₀ᵢ ∈ n₀, pᵢ ∈ p, pₗᵢ ∈ pₗ, tₗᵢ ∈ tₗ, cᵢ ∈ c, tᵣᵢ ∈ tᵣ, dᵢ ∈ d
        newest = monte_carlo(G, n, n₀ᵢ, shareprob(pᵢ), layerprob(pₗᵢ), tₗᵢ, conflevel(cᵢ), evac(tᵣᵢ), dᵢ; u, pb = false);
        DF.insertcols!.(newest, :n₀ => n₀ᵢ);
        DF.insertcols!.(newest, :p => pᵢ);
        DF.insertcols!.(newest, :pₗ => tuple(pₗᵢ));
        DF.insertcols!.(newest, :tₗ => tuple(tₗᵢ));
        DF.insertcols!.(newest, :c => cᵢ);
        DF.insertcols!.(newest, :tᵣ => tᵣᵢ);
        DF.insertcols!.(newest, :d => dᵢ);

        DF.append!.(data, newest);

        PM.next!(progress_bar);
    end

    data
end

namedtuple_to_string(nt::NamedTuple)::String = namedtuple_to_string(nt, keys(nt));
namedtuple_to_string(nt::NamedTuple, nt_keys)::String = join(map(k -> string(k) * " = " * string(nt[k]), nt_keys), ", ");

"""
Not needed for anything; a convenience function. Returns a subset of the key-value pairs in d where the key != x.
"""
except(d, x) = filter(i -> i.first != x, d);
except(d, x::Vector) = filter(i -> i.first ∉ x, d);

"""
Returns a subset of the dataframe `df` based on the dictionary `d`.
`d`: A dictionary of `key => [value]` pairs where `[value]` is the range of accepted values in the column `key`.
     Essentially an OR operation in `[value]` and an AND operation with the keys to determine which rows to keep.
"""
subsetresults(df, d) = DF.subset(df, map(k -> k => x -> x .∈ tuple(d[k]), collect(keys(d)))...);

draw_disseminate(ax, df; kwargs...) = draw_run(ax, df, :dissem; kwargs...);

function draw_run(ax, df, y; kwargs...)
    Makie.lines!(ax, df.t, df[!, y]; kwargs...);
end

function draw_layers(ax, df; kwargs...)
    Makie.lines!(ax, df.t, df.phone; label = "phone", kwargs...);
    Makie.lines!(ax, df.t, df.wom; label = "word of mouth", kwargs...);
    Makie.lines!(ax, df.t, df.sm; label = "social media", kwargs...);
    Makie.axislegend(ax; unique = true);
end

function draw_monte_carlo(ax, df, y = :dissem; sortby = [], opacity = .1)
    for dfᵢ ∈ DF.groupby(df, [:run, sortby...])
        draw_run(ax, dfᵢ, y; color = Colors.RGBA(0., 0., 0., opacity));
    end
end

function draw_sensitivity_analysis(ax, df, x::Symbol, y::Symbol = :dissem; kwargs...)
    agg_df = DF.combine(dfᵢ -> DF.combine(dfⱼ -> last(dfⱼ), DF.groupby(dfᵢ, :run)), DF.groupby(df, [:n₀, :p, :pₗ, :tₗ, :c, :tᵣ, :d]));
    Makie.scatter!(ax, agg_df[!, x], agg_df[!, y]; color = Colors.RGBA(0., 0., 0., 1.), markersize = 6, marker = 'o', kwargs...);
end

function draw_grid_monte_carlo(f, df, yx, y = :dissem; sortby = [], opacity = .1)
    gdf = DF.groupby(df, yx; sort = true);
    params = keys(gdf);
    titles = namedtuple_to_string.(NamedTuple.(params), tuple(yx));
    # Fortunately we won't have many plots, otherwise this would be very expensive
    y_len = (length ∘ unique)(getindex.(params, yx[1]));
    x_len = (length ∘ unique)(getindex.(params, yx[2]));

    axs = [];
    for i ∈ 1:y_len, j ∈ 1:x_len
        count = (i - 1) * x_len + j;
        push!(axs, Makie.Axis(f[i, j]; title = titles[count]));
    end
    for ax ∈ axs[(end - x_len + 1):end]
        ax.xlabel = "t";
    end
    for ax ∈ axs[1:x_len:end]
        ax.ylabel = "Nodes Informed";
    end
    for (ax, dfᵢ) ∈ zip(axs, gdf)
        draw_monte_carlo(ax, dfᵢ, y; sortby, opacity);
    end
end

function draw_row_sensitivity_analysis(f, df, x, row, y = :dissem; kwargs...)
    agg_df = DF.combine(dfᵢ -> DF.combine(dfⱼ -> last(dfⱼ), DF.groupby(dfᵢ, :run)), DF.groupby(df, [:n₀, :p, :pₗ, :tₗ, :c, :tᵣ, :d]));
    gdf = DF.groupby(agg_df, row);
    params = keys(gdf);
    titles = namedtuple_to_string.(NamedTuple.(params), tuple([row]));

    axs = [Makie.Axis(f[1, i]; xlabel = string(x), title = titles[i]) for i ∈ 1:length(params)];
    Makie.linkaxes!(axs...);
    axs[1].ylabel = "Total Nodes Informed";
    for (ax, dfᵢ) ∈ zip(axs, gdf)
        draw_sensitivity_analysis(ax, dfᵢ, x, y; kwargs...);
    end
end

function avgdeg(g)
    deg = LG.degree(g);
    sum(deg) / length(deg)
end

function coverage(g)
    deg = LG.degree(g);
    1 - count(isequal(0), deg) / length(deg)
end

save(filepath, df) = JLSO.save(filepath, :data => df);

load(filepath) = JLSO.load(filepath)[:data];

fig = Makie.Figure();
fig

#draw(SVG(joinpath("results", "test.svg")), gplot(g));
#Makie.save(joinpath("results", "filename.png"), fig);

end
