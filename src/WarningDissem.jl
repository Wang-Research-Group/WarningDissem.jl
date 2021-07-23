module WarningDissem

# Write your package code here.
using LightGraphs, MetaGraphs
using Distributions
using GLMakie, Colors
#using Compose, GraphPlot

#dba = barabasi_albert(1000, 2; is_directed=true)
#init_layers(n) = [barabasi_albert(n, 2), watts_strogatz(n, 4, 0.5), erdos_renyi(n, 2n)];


#=
g = grid((100, 100))
# Starting with a basic ER network
# 1001 nodes, .004 prob
#g = erdos_renyi(100001, .00004)
# Expected degree: 4
# Percolation threshold p: .25
p = .5

# Uniform distribution
d = Uniform()

# How to check if a stochastic event happens
happens(p) = rand(d) ≤ p;

# Doing this for now because the example's a single-layer network
init_layers(n) = [g];


get_neighbors(g) = map(x -> rand(d) ≤ p ? x : [], neighbors.(tuple(g), frontier));

n = 10000;
G = init_layers(n);

total_resultsˣ, total_resultsʸ = [], [];
highest_resultsˣ, highest_resultsʸ = [], [];

frontier = Set();
reached = Set();
step = 0;
resultsʸ = [];

function run(num_nodes)
    # These are the ones we're starting from
    nodes = [rand(1:10000) for _ ∈ 1:num_nodes];
    global frontier = Set(nodes);
    global reached = copy(frontier);
    global step = 0;
    push!(resultsʸ, length(reached));
    #println("Starting with node: ", node);

    while !isempty(frontier)
        global step += 1;
        # 3D vector; graph layers, frontier nodes, neighbors
        g_neighbors = get_neighbors.(G);
        talk_to = Iterators.flatten(Iterators.flatten(g_neighbors));
        #println(talk_to);
        global frontier = setdiff(Set(talk_to), reached);
        #println(frontier);
        union!(reached, talk_to);
        reached_len = length(reached);
        #println(step, ": ", reached_len);
        push!(resultsʸ, reached_len);
    end
end
=#

function run(G, p, d, n₀)
    # These are the nodes we're starting from
    nodes = rand(vertices(G[1]), n₀);
    frontier = Set(nodes);
    reached = copy(frontier);
    step = 0;
end

#=
function iter(num, num_starting)
    global total_resultsˣ = [];
    global total_resultsʸ = [];
    #global highest_resultsˣ = [];
    #global highest_resultsʸ = [];
    for i ∈ 1:num
        global resultsʸ = [];
        #println("Iteration ", i);
        run(num_starting)

        resultsˣ = 0:step
        resultsʸ = convert.(Int, resultsʸ)
        push!(total_resultsˣ, resultsˣ)
        push!(total_resultsʸ, resultsʸ)
        push!(highest_resultsˣ, p)
        push!(highest_resultsʸ, last(resultsʸ))
    end
end

for j ∈ 0:.01:1
    global p = j;
    println("p = ", p);
    iter(1000, 100);
end

#lines(total_resultsˣ[1], total_resultsʸ[1]; color = RGBA(0., 0., 0., .1));
#lines!.(total_resultsˣ[2:end], total_resultsʸ[2:end]; color = RGBA(0., 0., 0., .1));
highest_resultsˣ = convert.(Float32, highest_resultsˣ);
highest_resultsʸ = convert.(Int, highest_resultsʸ);
scatter(highest_resultsˣ, highest_resultsʸ; color = RGBA(0., 0., 0., 1.), markersize = 6);
current_figure()
=#

#draw(SVG("test.svg"), gplot(g));

end
