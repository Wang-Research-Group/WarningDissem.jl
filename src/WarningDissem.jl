module WarningDissem

# Write your package code here.
import LightGraphs, MetaGraphs
import Distributions
import DataFrames
import DataStructures
import CSV
import GLMakie, Colors
import ProgressMeter

const LG = LightGraphs;
const MG = MetaGraphs;
const Dist = Distributions;
const DF = DataFrames;
const DS = DataStructures;
const Makie = GLMakie;
const PM = ProgressMeter;

const Float = AbstractFloat;
const Range = AbstractRange;
#using Compose, GraphPlot

"""
Since Julia still doesn't have an `unzip()`...
"""
unzip(a) = map(x -> getfield.(a, x), (fieldnames ∘ eltype)(a));

function initnet!(G)
    # All vertex properties are set in the first layer only
    g = G[1];
    for v ∈ LG.vertices(g)
        neighbors = (unique ∘ Iterators.flatten ∘ broadcast)(LG.neighbors, G, v);
        # If we actually start using trust, this will have to be something else
        trust = zip(neighbors, (ones ∘ length)(neighbors)) |> Dict;
        MG.set_prop!(g, v, :state, NodeState(trust));
    end
end

function makenet()
    # [Phone, Word of Mouth, Social Media]
    wom, dists = begin
        coords = CSV.File(joinpath("data", "wom_coords.csv"); types = [Float64, Float64]) |> DF.DataFrame;
        coord_matrix = collect(Matrix(coords)');
        LG.euclidean_graph(coord_matrix; cutoff = 60.)
    end;
    wom = MG.MetaGraph(wom);

    n = LG.nv(wom);
    phone = LG.watts_strogatz(n, 10, .7) |> MG.MetaGraph;
    sm = LG.barabasi_albert(n, ceil(.374n) |> Int, 105) |> MG.MetaGraph;

    [phone, wom, sm]
end

"""
Since many of the parameters to `disseminate()` can take either a distribution or a value, we should
have something that samples the distribution if that's what's provided, or just returns the value.
NB: None of the results from this can be negative, so if that's something you need then either offset
    the results outside of this function or add another version that doesn't clamp it.
"""
sample(x::Dist.Distribution) = max(Dist.rand(x), 0);
sample(x) = max(x, 0);
sample(x::Dist.Distribution, n) = max.(Dist.rand(x, n), 0);
sample(x, n) = max.(fill(x, n), 0);

"""
`dₜ`: Time remaining until disaster
`tₗ`: Array of time it takes to share info on each layer
`conf`: Confidence in the info
"""
function shareprob(p)
    function (dₜ, tₗ, conf)
        conf * p * max(1 - (minimum ∘ broadcast)(sample, tₗ) / max(dₜ, 0), 0)
    end
end

function layerprob(base)
    function (dₜ, tₗ)
        @. unscaled = base * max(1 - sample(tₗ) / max(dₜ, 0), 0);
        unscaled ./ sum(unscaled)
    end
end

function conflevel(cₙ)
    function (informed, trust)
        # If informed by the broadcast, have the highest confidence level
        if ismissing(informed[1])
            return 1.
        end
        x = map(i -> trust[i], informed);
        clamp(sum(x) / (cₙ + 1), 0, 1)
    end
end

"""
`dₜ`: Time until the disaster occurs
    Range: [0, ∞) (actually could be passed in as a neg number, but clamp enforces this range for the calc)
`c`: Confidence in the information
    Range: [0, 1]

`tᵣ` is the max time that anyone would evac at; this is a linear scale to 0, so if 100% confident in the info
then a 50% chance to evac would be `tᵣ`/2 and a 100% chance would be 0.
"""
function evac(tᵣ)
    function (dₜ, c)
        c * (1 - clamp(dₜ / tᵣ, 0, 1))
    end
end

"""
Every node has an instance of this struct associated with it

`comm`: If I've started communicating already
`informed`: The list of nodes who have informed me
`trust`: A dictionary of how much I trust neighboring nodes
"""
mutable struct NodeState
    comm::Bool
    evac::Bool
    informed::Vector
    trust::Dict
end

NodeState(trust::Dict) = NodeState(false, false, [], trust);

"""
Properties of a communication between nodes

`t`: When the communication ends
`t₀`: When the communication starts
`src`: The node that transmits the information
`srcₗ`: The layer index of the communication (`Missing` if the communication is from the initial broadcast)
"""
struct Comm
    t::Float
    t₀::Float
    src
    srcₗ::Union{Int, Missing}
end

# Need these for the priority queue
Base.isless(a::Comm, b::Comm) = isless((a.t, a.t₀), (b.t, b.t₀));
Base.isequal(a::Comm, b::Comm) = isequal((a.t, a.t₀), (b.t, b.t₀));

"""
If you're given an array of times (`times`), and each time is from one node having some state change,
this figures out for each time how many nodes have had that state change to date.
Note that it assumes the times (`times`) are already sorted in increasing order.
"""
hist2cmf(times) = (unzip ∘ reverse ∘ unique)(x -> x[2], (reverse ∘ collect ∘ enumerate)(times));

"""
Parameter `u` is always the uniform distribution, but we pass it in so that the distribution doesn't
have to be recreated on every run of the dissemination simulation
"""
function disseminate(G::Vector, n₀::Int, p, pₗ, tₗ, c, r, d; u::Dist.Distribution = Dist.Uniform())
    # These are the nodes we're starting from
    nodes = Dist.sample(LG.vertices(G[1]), n₀; replace = false);

    events = DS.PriorityQueue();
    DS.enqueue!.(tuple(events), nodes, tuple(Comm(0, 0, missing, missing)));
    informed_times, evac_times = [], [];

    # Make sure all of the initial properties of the network are set
    initnet!(G);

    while !isempty(events)
        node, comm = DS.dequeue_pair!(events);
        t = comm.t;

        # This is a reference, so any changes will be made inside the graph as well
        state = MG.get_prop(G[1], node, :state);

        push!(state.informed, comm.src);

        # The node already decided to communicate
        if state.comm
            continue;
        end

        # If this is the first time the node is being informed, log it for the results
        if length(state.informed) == 1
            push!(informed_times, t);
        end

        conf = c(state.informed, state.trust);

        # If the node decides to evacuate
        if !state.evac && Dist.rand(u) ≤ (sample ∘ r)(d - t, conf)
            push!(evac_times, t);

            for neighbor ∈ LG.neighbors(G[2], node)
                # Remove from word-of-mouth only
                success = LG.rem_edge!(G[2], node, neighbor);
                if !success
                    error("Unable to remove node $node from word-of-mouth network (neighbor $neighbor)");
                end
            end

            state.evac = true;
        end

        # If the node decides to communicate to its contacts
        if Dist.rand(u) ≤ (sample ∘ p)(d - t, tₗ, conf)
            # Pick a layer based on the weights of `pₗ`
            layer = searchsortedfirst((cumsum ∘ broadcast)(sample, pₗ(d - t, tₗ)), Dist.rand(u));
            contacts = LG.neighbors(G[layer], node);

            # If reaching out on social media, contact everyone at once
            comm_contacts = if layer == 3
                zip(contacts, fill(sample(tₗ[layer]), length(contacts)))
            else
                # Randomly select a number of peers to contact
                # NB: The distribution `u` only works here because it's between 0-1
                num_contacting = ceil(Dist.rand(u) * length(contacts)) |> Int;
                contacting = Dist.sample(contacts, num_contacting; replace = false);
                zip(contacting, (cumsum ∘ sample)(tₗ[layer], length(contacting)))
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

    dissem, dissem_t = hist2cmf(informed_times);
    evac, evac_t = hist2cmf(evac_times);

    DF.DataFrame(t = dissem_t, dissem = dissem), DF.DataFrame(t = evac_t, evac = evac)
end

function monte_carlo(G::Vector, n::Int, n₀::Int, p, pₗ, tₗ, c, r, d; u::Dist.Distribution = Dist.Uniform(), pb = true)
    df, evacs = DF.DataFrame(), DF.DataFrame();

    progress_bar = PM.Progress(n; enabled = pb);

    for i ∈ 1:n
        newest_run, newest_evacs = disseminate(deepcopy(G), n₀, p, pₗ, tₗ, c, r, d; u = u);
        newest_run.run = fill(i, DF.nrow(newest_run));
        newest_evacs.run = fill(i, DF.nrow(newest_evacs));
        DF.append!(df, newest_run);
        DF.append!(evacs, newest_evacs);

        PM.next!(progress_bar);
    end

    df, evacs
end

function sensitivity_analysis(G::Vector, n::Int, n₀, p, tᵣ, tₗ, d; pb = true)
    df, evacs = DF.DataFrame(), DF.DataFrame();
    u = Dist.Uniform();

    progress_bar = PM.Progress(length(n₀) * length(p); enabled = pb);

    for n₀ᵢ ∈ n₀, pᵢ ∈ p, tᵣᵢ ∈ tᵣ
        newest_run, newest_evacs = monte_carlo(deepcopy(G), n, n₀ᵢ, shareprob(pᵢ), layerprob([.3, .7, 0.]), tₗ, conflevel(2), evac(tᵣᵢ), d; u = u, pb = false);
        dissem_len = DF.nrow(newest_run);
        evac_len = DF.nrow(newest_evacs);
        newest_run.n₀ = fill(n₀ᵢ, dissem_len);
        newest_run.p = fill(pᵢ, dissem_len);
        newest_run.tᵣ = fill(tᵣᵢ, dissem_len);
        newest_evacs.n₀ = fill(n₀ᵢ, evac_len);
        newest_evacs.p = fill(pᵢ, evac_len);
        newest_evacs.tᵣ = fill(tᵣᵢ, evac_len);

        DF.append!(df, newest_run);
        DF.append!(evacs, newest_evacs);

        PM.next!(progress_bar);
    end

    df, evacs
end

namedtuple_to_string(nt::NamedTuple)::String = namedtuple_to_string(nt, keys(nt));

namedtuple_to_string(nt::NamedTuple, nt_keys)::String = join(map(k -> string(k) * " = " * string(nt[k]), nt_keys), ", ");

function draw_disseminate(ax, df; kwargs...)
    Makie.lines!(ax, df.t, df.dissem; kwargs...);
end

function draw_monte_carlo(ax, df)
    for dfᵢ ∈ DF.groupby(df, :run)
        draw_disseminate(ax, dfᵢ; color = Colors.RGBA(0., 0., 0., .1));
    end
end

function draw_sensitivity_analysis(ax, df, x::Symbol)
    agg_df = DF.combine(dfᵢ -> DF.combine(dfⱼ -> last(dfⱼ), DF.groupby(dfᵢ, :run)), DF.groupby(df, [:n₀, :p, :tᵣ]));
    Makie.scatter!(ax, agg_df[!, x], agg_df.dissem; color = Colors.RGBA(0., 0., 0., 1.), markersize = 6);
end

function draw_grid_monte_carlo(f, df, yx)
    gdf = DF.groupby(df, yx; sort = true);
    params = keys(gdf);
    titles = namedtuple_to_string.(NamedTuple.(params), tuple(yx));
    # Fortunately we won't have many plots, otherwise this would be very expensive
    y_len = (length ∘ unique)(getindex.(params, yx[1]));
    x_len = (length ∘ unique)(getindex.(params, yx[2]));

    axs = [];
    for i ∈ 1:y_len, j ∈ 1:x_len
        count = (i - 1) * x_len + j;
        push!(axs, Makie.Axis(f[i, j], title = titles[count]));
    end
    for ax ∈ axs[(end - x_len + 1):end]
        ax.xlabel = "t";
    end
    for ax ∈ axs[1:x_len:end]
        ax.ylabel = "Nodes Informed";
    end
    for (ax, dfᵢ) ∈ zip(axs, gdf)
        draw_monte_carlo(ax, dfᵢ);
    end
end

function draw_row_sensitivity_analysis(f, df, x, row)
    agg_df = DF.combine(dfᵢ -> DF.combine(dfⱼ -> last(dfⱼ), DF.groupby(dfᵢ, :run)), DF.groupby(df, [:n₀, :p]));
    gdf = DF.groupby(agg_df, row);
    params = keys(gdf);
    titles = namedtuple_to_string.(NamedTuple.(params), tuple([row]));

    axs = [Makie.Axis(f[1, i], xlabel = string(x), title = titles[i]) for i ∈ 1:length(params)];
    axs[1].ylabel = "Total Nodes Informed";
    for (ax, dfᵢ) ∈ zip(axs, gdf)
        draw_sensitivity_analysis(ax, dfᵢ, x);
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

fig = Makie.Figure();
fig

#draw(SVG(joinpath("results", "test.svg")), gplot(g));
#Makie.save(joinpath("results", "filename.png"), fig);

end
