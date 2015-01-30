using Gadfly
using ProgressMeter

# Creates an array of length N where each entry is an independent realization
# of a complex Gaussian with Ez = Ezz = 0 and Ezz^* = 1.
function zGaussian(N=1)
   x, y = rand(N), rand(N)
   sqrt(-log(x)) .* exp(2im * pi * y)
end


function getRHS(nwells, T, KdJ)
   # Diagonal entries (0 for the action)
   center = (nwells + 1) / 2
   diags = push!(.5 * KdJ * ([1:nwells] - center).^2, 0)
   prop = -1im * diagm(diags)
   # Coupling between the wells
   coupls = push!(ones(nwells - 1) * T, 0)
   prop += 1im * (diagm(coupls, 1) + diagm(coupls, -1))

   (t, y) -> prop * y
end


function integrate_rk4(rhs, times, y0)
   y = zeros(typeof(y0[0]), (length(y0), length(times)))

   y[:, 1] = y0
   yrk = zeros(typeof(y0[0]), (length(y0), 4))

   for (i, t) in enumerate(times[2:end])
      dt = t - times[i]
      yrk[:, 1] = dt * rhs(t, y[:, i])
      yrk[:, 2] = dt * rhs(t + dt/2, y[:, i] + .5 * yrk[:, 1])
      yrk[:, 3] = dt * rhs(t + dt/2, y[:, i] + .5 * yrk[:, 2])
      yrk[:, 4] = dt * rhs(t + dt, y[:, i] + yrk[:, 3])

      y[:, i+1] = y[:, i] + 1/6 * (yrk[:, 1] + 2*yrk[:, 2] + 2*yrk[:, 3] + yrk[:, 4])
   end

   y
end

function main(timesteps, dt, nwells, realizations; T=2.0, KdJ=2.0)
   println("GO!")
   shift = convert(Int, floor((nwells - 1) // 2 % 2))
   even_w = filter(n -> isodd(n - shift), [1:nwells])
   odd_w = filter(n -> iseven(n - shift), [1:nwells])
   t = [1:timesteps] * dt
   rhs = getRHS(nwells, T, KdJ)
   norm = zeros(Complex{Float64}, timesteps)
   prog = Progress(realizations, 1)

   for n in [1:realizations]
      z1 = integrate_rk4(rhs, t, push!(zGaussian(nwells), 0))
      z2 = integrate_rk4(rhs, t, push!(zGaussian(nwells), 0))

      prefactor = prod(conj(z1[even_w, 1]) .* z2[even_w, 1])
      # Refactor as dot product
      sum_even = reshape(sum(z1[even_w, :] .* conj(z2[even_w, :]), 1), timesteps)
      sum_odd = reshape(sum(z1[odd_w, :] .* conj(z2[odd_w, :]), 1), timesteps)

      norm += prefactor * exp(sum_even + sum_odd) / realizations
      next!(prog)
   end

   println("DOne")
   t, norm
end


t, norm = main(100, 0.01, 5, 80000)
plot(x=t, y=real(norm), Geom.line)
