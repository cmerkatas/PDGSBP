function genericPDGSBP(x, maxiter, burnin, seed, filename = "/results")
  ### set seed and create a folder in the "savelocation" path to store results
  srand(seed)
  savelocation = string(pwd(), filename, "/seed$seed")
  mkpath(savelocation)


  for i in 1:1size(x)[1]
    writedlm(string(savelocation, "/x$i.txt"), x[i])
  end
  n = [length(x[j]) for j in 1:1:size(x)[1]]

  ### initialization
  NSAMPLES = length(n)
  epsilons = eye(Int64, NSAMPLES)
  idx = Int(maxiter/1000)
  times_elapsed = Array(Float64, idx)

  ### initialize clustering and geometric-slice variables
  ### each collection of data x[j] has its corresponding clustering variables d[j] and slice var N[j]
  # the delta[j] indicate the infinite mixture within the finite mixture each observation came from
  d = cell(NSAMPLES)
  N = cell(NSAMPLES)
  delta = cell(NSAMPLES)
  for j in 1:1:NSAMPLES
    d[j] = rand(DiscreteUniform(1,5),n[j])
    delta[j] = rand(Multinomial(1, 1/NSAMPLES * ones(NSAMPLES)), n[j])'
    N[j] = copy(d[j])
  end

  ### set up prior hyperparameters of the geometric stick breaking base measure G_0
  ### here we set as G_0 the independent NormalxGamma measure
  ### prior hyperparameter of the mean component
  mu0 = 0.0
  tau0 = 0.001
  sigma0 = sqrt(1.0 / tau0)

  ### prior hyperparameters of the precision components
  at = 0.001
  bt = 0.001

  ### initialize mixture selection probabilities and Dirichlet hyperprior alpha
  p = (1/NSAMPLES) * ones(NSAMPLES, NSAMPLES)
  alphajl = ones(NSAMPLES, NSAMPLES)
  P = cell(maxiter)

  ### Initialize geometric probability and hyperparameters of the conjugate beta prior
  z = 0.5
  az = 1.0
  bz = 1.0

  ### construct a symmetric matrix of random measures with pairwise dependence
  ### each element of the matrix is a Geometric stick breaking mixture
  Q = Matrix(NSAMPLES, NSAMPLES)
  M = maximum([maximum(N[r]) for r in 1:1:NSAMPLES])
  for j in 1:1:NSAMPLES
    for i in 1:1:j
      Q[i, j] = GSB(zeros(M), ones(M), z, zeros(M), zeros(M), zeros(maxiter))
    end
  end
  Q = Symmetric(Q)

  ### store clusters and samples from predictives in arrays
  fclusters = Array(Int64, NSAMPLES, maxiter)
  pred = Array(Float64, NSAMPLES, maxiter - burnin)

  println("Starting Gibbs...")
  start_time = tic()

  for its in 1:1:maxiter
    M = maximum([maximum(N[r]) for r in 1:1:NSAMPLES])

    for l in 1:1:NSAMPLES
      for j in 1:1:l
        Q[j,l].mu = zeros(M)
        Q[j,l].w = zeros(M)
        Q[j,l].W = zeros(M)
        for k in 1:1:M
          Q[j,l].w[k] = Q[j,l].z * (1 - Q[j,l].z)^(k - 1)

          ### sampling of locations for the idiosyncratic
          # sampling of the means
          ksij, mj, gj = 0.0, 0.0, 0.0
          if j .== l
            for i in 1:1:n[j]
              if (d[j][i] .== k) & (delta[j][i,l] .== 1)
                mj += 1
                ksij += x[j][i]
              end
            end
          else # sample the locations for the common parts between densities fⱼ and fl
            for i in 1:1:n[j]
              if (d[j][i] .== k) & (delta[j][i,l] .== 1)
                mj += 1
                ksij += x[j][i]
              end
            end
            for i in 1:1:n[l]
              if (d[l][i] .== k) & (delta[l][i,j] .== 1)
                mj += 1
                ksij += x[l][i]
              end
            end
          end # end if i==j
          meanstar = Q[j,l].lambda[k] * ksij / (tau0 + Q[j,l].lambda[k] * mj)
          varstar = 1.0 / (tau0 + Q[j,l].lambda[k] * mj)
          Q[j,l].mu[k] = rand(Normal(meanstar, varstar^0.5))

          ### sampling of location for the idiosyncratic parts
          # sampling of precisions
          if j .== l
            for i in 1:1:n[j]
              if (d[j][i] .== k) & (delta[j][i,l] .== 1)
                gj += (x[j][i] - Q[j,l].mu[k])^2
              end
            end
          else # sample locations for the common parts
            for i in 1:1:n[j]
              if (d[j][i] .== k) & (delta[j][i,l] .== 1)
                gj += (x[j][i] - Q[j,l].mu[k])^2
              end
            end
            for i in 1:1:n[l]
              if (d[l][i] .== k) & (delta[l][i,j] .== 1)
                gj += (x[l][i] - Q[j,l].mu[k])^2
              end
            end
          end # if j==l
          alphastar = at + 0.5 * mj
          betastar = bt + 0.5 * gj
          Q[j,l].lambda[k] = rand(Gamma(alphastar, betastar^(-1)))

        end # k ind
      end
    end

    for j in 1:1:NSAMPLES
      d[j], delta[j] = sampleClustAllocs(Q[j,:], x[j], p[j,:], N[j]) # sample clustering variables
      fclusters[j, its] = uniqueClusters(d[j], delta[j]) # unique clusters on j density at its iteration
      N[j] = sampleGeomSlice(Q[j,:], d[j], delta[j])  # sample geometric slice variables
      p[j, :] = sampleSelectionProbabilities(alphajl[j, :], delta[j]) # update the mixture selection probabilities
    end
    P[its] = copy(p)



    if its > burnin
      for j in 1:1:NSAMPLES
        ### density estimation
        pred[j, its - burnin] = samplePredictive(Q[j,:], p[j,:], mu0, sigma0, at, bt) # sample from the predictive density
      end
    end

    ### update geometric probability
    for l in 1:1:NSAMPLES
      for j in 1:1:l
        if j == l # for the idiosyncratic part
          Q[j,l].z = rand(Beta(az + 2 * sum(delta[j][:, l].==1), bz + sum(N[j][delta[j][:,l].==1]) - sum(delta[j][:, l].==1)))
        else  # and for the common parts
          Q[j,l].z = rand(Beta(az + 2 * (sum(delta[j][:, l].==1) + sum(delta[l][:, j].==1)),
             bz + (sum(N[j][delta[j][:,l].==1]) + sum(N[l][delta[l][:,j].==1])) - (sum(delta[j][:, l].==1) + sum(delta[l][:,j].==1))))
        end
        Q[j,l].Z[its] = Q[j,l].z
      end
    end

    ### adjust array of precision for next iteration
    if maximum([maximum(N[r]) for r in 1:1:NSAMPLES]) > M
      for l in 1:1:NSAMPLES
        for j in 1:1:l
          Q[j, l].lambda = [Q[j, l].lambda;ones(maximum([maximum(N[r]) for r in 1:1:NSAMPLES]) - M)]
        end
      end
    end

    if its%1000 == 0
      end_time = toc()
      times_elapsed[Int(its/1000)] = end_time
      println("MCMC iterations: $its", " in ", end_time, " seconds")
      start_time = tic()
    end

  end ### end gibbs
  writedlm(string(savelocation, "/predictives.txt"), pred')
  writedlm(string(savelocation, "/clusters.txt"), fclusters')
  writedlm(string(savelocation, "/times.txt"), times_elapsed)
  writedlm(string(savelocation, "/Prob.txt"), P)


  for l in 1:1:NSAMPLES
    for j in 1:1:l
      writedlm(string(savelocation, "/L$l$j.txt"), Q[j,l].Z)
    end
  end


  return pred, P, fclusters, times_elapsed
end
