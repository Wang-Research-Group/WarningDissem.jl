module WarningDissem

# Write your package code here.
using LightGraphs, MetaGraphs
using Distributions
using DataFrames
using GLMakie, Colors
using ProgressMeter

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

get_neighbors(g, frontier, p, d) = map(x -> rand(d) ≤ p ? x : [], neighbors.(tuple(g), frontier));

disseminate(G::Vector, n₀::Int, p::Float)::DataFrame = disseminate(G, n₀, p, Uniform());

function disseminate(G::Vector, n₀::Int, p::Float, d::Distribution)::DataFrame
    # These are the nodes we're starting from
    nodes = sample(vertices(G[1]), n₀; replace = false);

    frontier = Set(nodes);
    reached = copy(frontier);
    t = 0;
    dissem_total::Vector{Int} = [];

    while !isempty(frontier)
        push!(dissem_total, length(reached));
        t += 1;

        # 3D vector; graph layers, frontier nodes, neighbors
        G_neighbors = get_neighbors.(G, tuple(frontier), p, tuple(d));
        talked_to = Iterators.flatten(Iterators.flatten(G_neighbors));
        frontier = setdiff(Set(talked_to), reached);
        union!(reached, talked_to);
    end

    DataFrame(n₀ = fill(n₀, t), p = fill(p, t), t = 0:(t - 1), dissem = dissem_total)
end

monte_carlo(G::Vector, n::Int, n₀::Int, p::Float; pb = true) = monte_carlo(G, n, n₀, p, Uniform(); pb = pb);

function monte_carlo(G::Vector, n::Int, n₀::Int, p::Float, d::Distribution; pb = true)::DataFrame
    df = DataFrame();

    progress_bar = Progress(n; enabled = pb);

    for i ∈ 1:n
        newest_run = disseminate(G, n₀, p, d);
        newest_run.run = fill(i, nrow(newest_run));
        append!(df, newest_run);

        next!(progress_bar);
    end

    df
end

function sensitivity_analysis(G::Vector, n::Int, n₀, p; pb = true)::DataFrame
    df = DataFrame();
    d = Uniform();

    progress_bar = Progress(length(n₀) * length(p); enabled = pb);

    for n₀ᵢ ∈ n₀, pᵢ ∈ p
        append!(df, monte_carlo(G, n, n₀ᵢ, pᵢ, d; pb = false));

        next!(progress_bar);
    end

    df
end

namedtuple_to_string(nt::NamedTuple)::String = namedtuple_to_string(nt, keys(nt));

namedtuple_to_string(nt::NamedTuple, nt_keys)::String = join(map(k -> string(k) * " = " * string(nt[k]), nt_keys), ", ");

function draw_disseminate(ax, df; kwargs...)
    lines!(ax, df.t, df.dissem; kwargs...);
end

function draw_monte_carlo(ax, df)
    for dfᵢ ∈ groupby(df, :run)
        draw_disseminate(ax, dfᵢ; color = RGBA(0., 0., 0., .1));
    end
end

function draw_sensitivity_analysis(ax, df, x::Symbol)
    agg_df = combine(dfᵢ -> combine(dfⱼ -> last(dfⱼ), groupby(dfᵢ, :run)), groupby(df, [:n₀, :p]));
    scatter!(ax, agg_df[!, x], agg_df.dissem; color = RGBA(0., 0., 0., 1.), markersize = 6);
end

function draw_grid_monte_carlo(f, df, yx)
    gdf = groupby(df, yx; sort = true);
    params = keys(gdf);
    titles = namedtuple_to_string.(NamedTuple.(params), tuple(yx));
    # Fortunately we won't have many plots, otherwise this would be very expensive
    y_len = length(unique(getindex.(params, yx[1])));
    x_len = length(unique(getindex.(params, yx[2])));

    axs = [];
    for i ∈ 1:y_len, j ∈ 1:x_len
        count = (i - 1) * x_len + j;
        push!(axs, Axis(f[i, j], title = titles[count]));
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
    agg_df = combine(dfᵢ -> combine(dfⱼ -> last(dfⱼ), groupby(dfᵢ, :run)), groupby(df, [:n₀, :p]));
    gdf = groupby(agg_df, row);
    params = keys(gdf);
    titles = namedtuple_to_string.(NamedTuple.(params), tuple([row]));

    axs = [Axis(f[1, i], xlabel = string(x), title = titles[i]) for i ∈ 1:length(params)];
    axs[1].ylabel = "Total Nodes Informed";
    for (ax, dfᵢ) ∈ zip(axs, gdf)
        draw_sensitivity_analysis(ax, dfᵢ, x);
    end
end

G = [grid((100, 100))];

fig = Figure();
fig

#draw(SVG("test.svg"), gplot(g));

end
