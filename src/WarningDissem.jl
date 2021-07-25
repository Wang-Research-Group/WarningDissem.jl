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
    nodes = rand(vertices(G[1]), n₀);

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

function monte_carlo(G::Vector, n::Int, n₀::Int, p::Float; pb = true)::DataFrame
    d = Uniform();
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

    progress_bar = Progress(length(n₀) * length(p); enabled = pb);

    for n₀ᵢ ∈ n₀, pᵢ ∈ p
        append!(df, monte_carlo(G, n, n₀ᵢ, pᵢ; pb = false));

        next!(progress_bar);
    end

    df
end

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

function draw_grid_monte_carlo(f::Figure, G::Vector, n::Int, n₀, p, x::Symbol)
    params = [];
    dfs = [];
    x_len, y_len = 0, 0;
    if x == :p
        x_len = length(n₀);
        y_len = length(p);
        for n₀ᵢ ∈ n₀, pᵢ ∈ p
            push!(params, (n₀ᵢ, pᵢ));
            push!(dfs, monte_carlo(G, n, n₀ᵢ, pᵢ; pb = false));
        end
    else
        x_len = length(p);
        y_len = length(n₀);
        for pᵢ ∈ p, n₀ᵢ ∈ n₀
            push!(params, (n₀ᵢ, pᵢ));
            push!(dfs, monte_carlo(G, n, n₀ᵢ, pᵢ; pb = false));
        end
    end
    axs = [];
    for i ∈ 1:x_len, j ∈ 1:y_len
        n₀, p = params[(i - 1) * y_len + j];
        push!(axs, Axis(f[i, j], xlabel = "t", title = "n₀ = $n₀, p = $p"));
    end
    for (ax, df) ∈ zip(axs, dfs)
        draw_monte_carlo(ax, df);
    end
end

G = [grid((100, 100))];

fig = Figure();
fig

#draw(SVG("test.svg"), gplot(g));

end
