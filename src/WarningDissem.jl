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

#dba = barabasi_albert(1000, 2; is_directed=true)
#init_layers(n) = [barabasi_albert(n, 2), watts_strogatz(n, 4, 0.5), erdos_renyi(n, 2n)];


# Starting with a basic ER network
# 1001 nodes, .004 prob
#g = erdos_renyi(100001, .00004)
# Expected degree: 4
# Percolation threshold p: .25
#p = .5

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
        LG.euclidean_graph(coord_matrix; cutoff = 50.)
    end;
    wom = MG.MetaGraph(wom);

    n = LG.nv(wom);
    phone = LG.watts_strogatz(n, 4, .5) |> MG.MetaGraph;
    sm = LG.barabasi_albert(n, 2) |> MG.MetaGraph;

    [phone, wom, sm]
end

"""
Since many of the parameters to `disseminate()` can take either a distribution or a value, we should
have something that samples the distribution if that's what's provided, or just returns the value
"""
sample(x::Dist.Distribution) = Dist.rand(x);
sample(x) = x;
sample(x::Dist.Distribution, n) = Dist.rand(x, n);
sample(x, n) = fill(x, n);

"""
`dₜ`: Time remaining until disaster
`tₗ`: Array of time it takes to share info on each layer
`conf`: Confidence in the info
"""
function shareprob(p)
    function (dₜ, tₗ, conf)
        # TODO: Actually use `dₜ` and `tₗ`
        conf * p * (1 + 0 * minimum(tₗ) / dₜ)
    end
end

function layerprob()
    function (dₜ, tₗ)
        # TODO: Actually use `dₜ` and `tₗ`
        [.3 + 0 * tₗ[1] / dₜ, .4 + 0 * tₗ[2] / dₜ, .3 + 0 * tₗ[3] / dₜ]
    end
end

function conflevel()
    function (informed, trust)
        # If informed by the broadcast, have the highest confidence level
        if ismissing(informed[1])
            return 1.
        end
        x = map(i -> trust[i], informed);
        clamp(sum(x) / 2, 0, 1)
    end
end

"""
`dₜ`: Time until the disaster occurs
    Range: [0, ∞) (actually could be passed in as a neg number, but clamp enforces this range for the calc)
`c`: Confidence in the information
    Range: [0, 1]

`σ` is the max time that anyone would evac at; this is a linear scale to 0, so if 100% confident in the info
then a 50% chance to evac would be `σ`/2 and a 100% chance would be 0.
"""
function evac(σ)
    function (dₜ, c)
        c * (1 - clamp(dₜ / σ, 0, 1))
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
hist_to_cmf(times) = (unzip ∘ reverse ∘ unique)(x -> x[2], (reverse ∘ collect ∘ enumerate)(times));

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

    dissem, dissem_t = hist_to_cmf(informed_times);
    evac, evac_t = hist_to_cmf(evac_times);

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

function sensitivity_analysis(G::Vector, n::Int, n₀, p, σ, tₗ, d; pb = true)
    df, evacs = DF.DataFrame(), DF.DataFrame();
    u = Dist.Uniform();

    progress_bar = PM.Progress(length(n₀) * length(p); enabled = pb);

    for n₀ᵢ ∈ n₀, pᵢ ∈ p, σᵢ ∈ σ
        newest_run, newest_evacs = monte_carlo(deepcopy(G), n, n₀ᵢ, shareprob(pᵢ), layerprob(), tₗ, conflevel(), evac(σᵢ), d; u = u, pb = false);
        dissem_len = DF.nrow(newest_run);
        evac_len = DF.nrow(newest_evacs);
        newest_run.n₀ = fill(n₀ᵢ, dissem_len);
        newest_run.p = fill(pᵢ, dissem_len);
        newest_run.σ = fill(σᵢ, dissem_len);
        newest_evacs.n₀ = fill(n₀ᵢ, evac_len);
        newest_evacs.p = fill(pᵢ, evac_len);
        newest_evacs.σ = fill(σᵢ, evac_len);

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
    agg_df = DF.combine(dfᵢ -> DF.combine(dfⱼ -> last(dfⱼ), DF.groupby(dfᵢ, :run)), DF.groupby(df, [:n₀, :p, :σ]));
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

G = [LG.grid((100, 100)) |> MG.MetaGraph];

fig = Makie.Figure();
fig

#draw(SVG(joinpath("results", "test.svg")), gplot(g));
#Makie.save(joinpath("results", "filename.png"), fig);

end
