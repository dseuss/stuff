module hmm
implicit none
public

contains

! Computes the (unscaled) forward variables of the HMM defined by
!
!     alpha_0[i] = init[i] * mko[obs[0], i]
!     alpha_t+1[i] = sum_j mkh[ij] * alpha_t[j] * mko[obs[t+1], i]
!
! where alpha is rescaled after each step alpha_t = alpha_t * c_t
! with the scaling variable
!
!     c_t = 1 / (sum_i alpha_t[i])
!
! :returns: (len(obs), len(mkh)) array, (len(obs)) array;
!     forward and scaling variables
subroutine get_forward(n_hid, n_obs, steps, mkh, mko, init, obs, forward, &
         scaling)
   implicit none
   integer, intent(in) :: n_hid
   integer, intent(in) :: n_obs
   integer, intent(in) :: steps
   real*8, intent(in)  :: mkh(n_hid, n_hid)
   real*8, intent(in)  :: mko(n_obs, n_hid)
   real*8, intent(in)  :: init(n_hid)
   integer, intent(in) :: obs(steps)
   real*8, intent(out) :: forward(steps, n_hid)
   real*8, intent(out) :: scaling(steps)

   integer :: t
   forward(1, :) = init * mko(obs(1), :)
   scaling(1) = to_scale(forward(1, :))

   do t = 1, steps - 1
      forward(t+1, :) = matmul(mkh, forward(t, :)) * mko(obs(t+1), :)
      scaling(t+1) = to_scale(forward(t+1, :))
   end do

contains

   function to_scale(forward_t) result(scaling_t)
      implicit none
      real*8, intent(inout) :: forward_t(:)
      real*8                :: scaling_t

      scaling_t = 1. / sum(forward_t)
      forward_t = forward_t * scaling_t
   end function to_scale

end subroutine get_forward


! Computes the (unscaled) backward variables of the HMM defined by
!
!    beta_T-1[i] = 1
!    beta_t[i] = sum_j mko[obs[t+1], j] beta_t+1[j] mkh[ji]
!
! where T = len(obs)
!
! :returns: (len(obs), len(mkh)) array; backward variables
subroutine get_backward(n_hid, n_obs, steps, mkh, mko, obs, scaling, backward)
   implicit none
   integer, intent(in) :: n_hid
   integer, intent(in) :: n_obs
   integer, intent(in) :: steps
   real*8, intent(in)  :: mkh(n_hid, n_hid)
   real*8, intent(in)  :: mko(n_obs, n_hid)
   integer, intent(in) :: obs(steps)
   real*8, intent(in)  :: scaling(steps)
   real*8, intent(out) :: backward(steps, n_hid)

   integer :: t
   backward(steps, :) = scaling(steps)

   do t = steps - 1, 1, -1
      backward(t, :) = matmul(mko(obs(t+1), :) * backward(t+1, :), mkh) * scaling(t)
   end do

end subroutine get_backward


subroutine iterate(n_hid, n_obs, steps, mkh, mko, init, obs, &
         mkh_n, mko_n, init_n)
   implicit none
   integer, intent(in) :: n_hid
   integer, intent(in) :: n_obs
   integer, intent(in) :: steps
   real*8, intent(in)  :: mkh(n_hid, n_hid)
   real*8, intent(in)  :: mko(n_obs, n_hid)
   real*8, intent(in)  :: init(n_hid)
   integer, intent(in) :: obs(steps)
   real*8, intent(out) :: mkh_n(n_hid, n_hid)
   real*8, intent(out) :: mko_n(n_obs, n_hid)
   real*8, intent(out) :: init_n(n_hid)

   real*8 forward(steps, n_hid), backward(steps, n_obs), &
         scaling(steps), gamma(steps, n_hid), gamma_sum(n_hid), &
         xi_t(n_hid, n_hid)
   integer t, i, j

   call get_forward(n_hid, n_obs, steps, mkh, mko, init, obs, forward, &
         scaling)
   call get_backward(n_hid, n_obs, steps, mkh, mko, obs, scaling, backward)

   do t = 1, steps
      gamma(t, :) = forward(t, :) * backward(t, :)
      gamma(t, :) = gamma(t, :) / sum(gamma(t, :))
   end do
   gamma_sum = sum(gamma(1:steps - 1, :), 1)

   mkh_n = 0.
   do t = 1, steps - 1
      do i = 1, n_hid
         do j = 1, n_hid
            xi_t(j, i) = forward(t, i) * mkh(j, i) * mko(obs(t+1), j) &
                  * backward(t+1, j)
         end do
      end do
      mkh_n = mkh_n + (xi_t / sum(xi_t))
   end do
   do i = 1, n_hid
      mkh_n(:, i) = gamma_sum(i)
   end do

   gamma_sum = gamma_sum + gamma(steps, :)
   mko_n = 0.
   do t = 1, steps
      mko_n(obs(t), :) = mko_n(obs(t), :) + gamma(t, :)
   end do
   do j = 1, n_hid
      mko_n(:, j) = mko_n(:, j) / gamma_sum(j)
   end do

   init_n = gamma(1, :)
end subroutine iterate
! gamma = forward * backward
! gamma /= np.sum(gamma, axis=-1)[:, None]
! xi = forward[:-1, None, :] * mkh[None, :, :] \
!    * mko[obs[1:], :, None] * backward[1:, :, None]
! xi /= np.sum(xi, axis=(1, 2))[:, None, None]

! init_n = gamma[0]
! mkh_n = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0, keepdims=True)
! mko_n = np.empty((len(mko), len(mkh)))
! gamma_sum = np.sum(gamma, axis=0)
! for k in range(len(mko)):
!    mko_n[k, :] = np.sum(gamma[obs == k], axis=0) / gamma_sum

! return mkh_n, mko_n, init_n
end module hmm
