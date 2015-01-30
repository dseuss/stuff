module hmm

type HMM{T}
   n_hid
   n_obs
   a :: Array{T,2}
   b :: Array{T,2}
   pi :: Array{T,1}
end


function HMM(n_hid, n_obs)
   HMM(n_hid, n_obs, normalize_stoch_map(rand((n_hid, n_hid))),
       normalize_stoch_map(rand((n_obs, n_hid))),
       normalize_stoch_map(rand(n_hid)))
end


function get_forward(m, obs)

   function rescale!(x)
      fac = 1. / sum(x)
      x[:] *= fac
      fac
   end

   steps = size(obs)
   scale = Array(eltype(m.a), steps)
   alpha = Array(eltype(m.a), (steps, m.n_hid))

   alpha[1, :] = m.pi .* m.b[obs[1], :]
   scale[1] = rescale!(alpha[1, :])

   for t in 2:steps
      alpha[t, :] = (m.a * alpha[t-1, :]) .* m.b[obs[t], :]
      scale[t] = rescale!(alpha[t, :])
   end

   alpha, scale
end


function get_backward(m, obs, scale)
   steps = size(obs)
   beta = Array(eltype(m.a), (steps, m.n_hid))
   beta[steps, :] = scale[steps]

   for t in steps-1:-1:1
      beta[t, :] = scale[t] * (m.b[obs[t+1], :] .* beta[t+1, :]) * m.a
   end
   beta
end


function iterate(m, obs)
   alpha, scale = get_forward(m, obs)
   beta = get_backward(m, obs, scale)

   gamma = alpha .* beta
   gamma /= broadcast(/, gamma, sum(gamma, 2))

   mn =
end

function normalize_stoch_map(a)
   normalization = sum(a, 1)
   broadcast(/, a, normalization)
end


end
