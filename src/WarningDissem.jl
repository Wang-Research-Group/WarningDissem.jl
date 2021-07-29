module WarningDissem

# Write your package code here.
import LightGraphs, MetaGraphs
import Distributions
import DataFrames
import DataStructures
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

# Since Julia still doesn't have an `unzip()`...
unzip(a) = map(x -> getfield.(a, x), fieldnames(eltype(a)));

get_neighbors(g, frontier, p, d) = map(x -> rand(d) ≤ p ? x : [], LG.neighbors.(tuple(g), frontier));

disseminate(G::Vector, n₀::Int, p::Float)::DF.DataFrame = disseminate(G, n₀, p, Dist.Uniform());

function disseminate(G::Vector, n₀::Int, p::Float, d::Dist.Distribution)::DF.DataFrame
    # These are the nodes we're starting from
    nodes = Dist.sample(LG.vertices(G[1]), n₀; replace = false);

    events = DS.PriorityQueue();
    DS.enqueue!.(tuple(events), nodes, 0);
    done = Set();
    t = [];

    while !isempty(events)
        node, time = DS.dequeue_pair!(events);
        push!(t, time);

        # If the node decides to communicate to its neighbors
        if rand(d) ≤ p
            # Currently assuming communication on all layers at once
            neighbors = Iterators.flatten(LG.neighbors.(G, node));
            # For testing purposes, spend between 1-5 timesteps communicating to a neighbor
            for neighbor in neighbors
                if neighbor ∉ done && neighbor ∉ keys(events)
                    DS.enqueue!(events, neighbor, time + rand(1:5));
                end
            end
        end

        push!(done, node);
    end

    # Yes, I know this looks complicated. Let me break it down:
    # If you're given an array of times (`t`), and each time is from one node being informed,
    # this figures out for each time how many nodes have been informed to date.
    # Note that it assumes the times (`t`) are already sorted in increasing order.
    dissem, unique_t = unzip(reverse(unique(x -> x[2], reverse(collect(enumerate(t))))));

    DF.DataFrame(n₀ = fill(n₀, length(dissem)), p = fill(p, length(dissem)), t = unique_t, dissem = dissem)
end

monte_carlo(G::Vector, n::Int, n₀::Int, p::Float; pb = true) = monte_carlo(G, n, n₀, p, Dist.Uniform(); pb = pb);

function monte_carlo(G::Vector, n::Int, n₀::Int, p::Float, d::Dist.Distribution; pb = true)::DF.DataFrame
    df = DF.DataFrame();

    progress_bar = PM.Progress(n; enabled = pb);

    for i ∈ 1:n
        newest_run = disseminate(G, n₀, p, d);
        newest_run.run = fill(i, DF.nrow(newest_run));
        DF.append!(df, newest_run);

        PM.next!(progress_bar);
    end

    df
end

function sensitivity_analysis(G::Vector, n::Int, n₀, p; pb = true)::DF.DataFrame
    df = DF.DataFrame();
    d = Dist.Uniform();

    progress_bar = PM.Progress(length(n₀) * length(p); enabled = pb);

    for n₀ᵢ ∈ n₀, pᵢ ∈ p
        append!(df, monte_carlo(G, n, n₀ᵢ, pᵢ, d; pb = false));

        PM.next!(progress_bar);
    end

    df
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
    agg_df = DF.combine(dfᵢ -> DF.combine(dfⱼ -> last(dfⱼ), DF.groupby(dfᵢ, :run)), DF.groupby(df, [:n₀, :p]));
    Makie.scatter!(ax, agg_df[!, x], agg_df.dissem; color = Colors.RGBA(0., 0., 0., 1.), markersize = 6);
end

function draw_grid_monte_carlo(f, df, yx)
    gdf = DF.groupby(df, yx; sort = true);
    params = keys(gdf);
    titles = namedtuple_to_string.(NamedTuple.(params), tuple(yx));
    # Fortunately we won't have many plots, otherwise this would be very expensive
    y_len = length(unique(getindex.(params, yx[1])));
    x_len = length(unique(getindex.(params, yx[2])));

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

G = [LG.grid((100, 100))];

fig = Makie.Figure();
fig

#draw(SVG("test.svg"), gplot(g));

end
