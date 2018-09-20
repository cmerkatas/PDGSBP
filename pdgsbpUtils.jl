### construct a type of GSB mixture
type GSB
  mu::Vector{Float64}
  lambda::Vector{Float64}
  z::Float64
  w::Vector{Float64}
  W::Vector{Float64}
  Z::Vector{Float64}
end

### function for the generation of length(sample_sizes) data sets coming frmo mixture densities
### with means and stds given by means and sigmas. The mixture selection probabilities are specified in mixprobs
function toy_data(sample_sizes::Array{Int64, 1}, means::Array{Float64, 2}, sigmas::Array{Float64, 2}, mixprobs = Array{Float64, 2}, seed=1)
  srand(seed)
  m = length(sample_sizes)
  cumprobs = cumsum(mixprobs, dims=2)
  x = fill(Float64[], m)
  for j in 1:1:m
    x[j] = zeros(sample_sizes[j])
    for i in 1:1:sample_sizes[j]
      u = rand()
      for l in 1:1:m
        if u < cumprobs[j,l]
          x[j][i] = rand(Normal(means[j,l], sigmas[j,l]))
          break
        end
      end
    end
  end
  return x
end

### sample the clustering variables in density f_j
function sampleClustAllocs(R::Array{Any, 2}, y, pr::Array{Float64, 2}, S::Array{Int64, 1})
  n = length(y)
  m = length(pr)
  clusters = Array(Int64, n)
  allocations = zeros(Int64, n, m)

  for i in 1:1:n
    nc = 0.0
    for l in 1:1:m
      for j in 1:1:S[i]
        nc += pr[l] * R[l].lambda[j]^0.5 * exp(-0.5 * R[l].lambda[j] * (y[i] - R[l].mu[j])^2)
      end
    end
    prob = 0.0
    r = rand()
    flag = false
    for l in 1:1:m
      for j in 1:1:S[i]
        prob += pr[l] * R[l].lambda[j]^0.5 * exp(-0.5 * R[l].lambda[j] * (y[i] - R[l].mu[j])^2) / nc
        if r < prob
          clusters[i] = j
          allocations[i, l] = 1
          flag = true
          break
        end
      end
      if flag
        break
      end
    end
  end
  return clusters, allocations
end

### function for simulating truncated geometric random variables
function stgeornd(p::Float64, k::Int64)
  z::Int64
  z = floor(log(rand(Uniform(0.0,1.0))) / log(1 - p)) + k
  return z
end

### function for the sampling of geometric slice variables N
function sampleGeomSlice(R::Array{Any, 2}, clusters::Array{Int64, 1}, allocations::Array{Int64, 2})
  n = length(clusters)
  s = zeros(Int64, n)
  for i in 1:1:n
    for l in 1:1:size(allocations, 2)
      if allocations[i,l].==1
        s[i] = stgeornd(R[l].z, clusters[i])
        break
      end
    end
  end
  return s
end

### simulates from the Dirichlet posterior of the selection probabilities
function sampleSelectionProbabilities(a::Array{Float64, 2}, alloc::Array{Int64, 2})
  p = Array(Float64, size(alloc, 2))
  adir = zeros(1, size(alloc, 2))
  for i in 1:1:size(alloc, 1)
    for l in 1:1:size(alloc, 2)
      if alloc[i, l] == 1
        adir[l] += 1.0
      end
    end
  end
  p = rand(Dirichlet(vec(a + adir)))'
  return p
end

### function for the calculation of the number of unique clusters in group j
function uniqueClusters(clusters::Array{Int64, 1}, allocations::Array{Int64, 2})
  ucl = 0
  l = size(allocations, 2)
  for i in 1:1:l
    ucl += length(unique(clusters[allocations[:,i].==1]))
  end
  return ucl
end

### density estimation with sampling from the predictive
function samplePredictive(R::Array{Any, 2}, pr::Array{Float64, 2}, μ, σ, α, β)
  m = length(pr)
  Pr = cumsum(pr)
  xp = rand(Normal(rand(Normal(μ, σ)), sqrt(1 / rand(Gamma(α, 1 / β)))))
  for l in 1:1:m
    R[l].W = cumsum(R[l].w)
    rc = rand()
    rf = rand()
    if rc < Pr[l]
      for i in 1:1:length(R[l].W)
        if rf < R[l].W[i]
          xp = rand(Normal(R[l].mu[i], sqrt(1 / R[l].lambda[i])))
          break
        end
      end
      break
    end
  end
  return xp
end
