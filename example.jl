using Distributions

include("pdgsbpUtils.jl")
include("genericPDGSBP.jl")
include("pddpUtils.jl")
include("genericPDDP.jl")

x = toy_data([100, 50, 100], [-30 -20 -10;-20 0. 20;-30 10 20], ones(3,3), 1/3 * ones(3,3))


(maxiter, burnin, seed) = (150000, 20000, 1)
precompile(genericPDGSBP, (maxiter, burnin, seed))
precompile(genericPDDP, (maxiter, burnin, seed))

# run the PDGSBP sampler
@time predictive, P, clusters, times = genericPDGSBP(x, maxiter, burnin, seed, "/example_res")
