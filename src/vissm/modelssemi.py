import jax
import jax.numpy as jnp
import rodeo.kalmantv
import equinox as eqx
from vissm.block_tridiag import *
from vissm.models import NN_Mean, NN_Lower, NN_Diag, RNN, BIRNN
from vissm.utils import smooth_sim

# archer model

class ArcherModel:
    def __init__(self, n_state):
        self.n_state = n_state
    
    def _theta_y_meas(self, params, y_meas):
        self.n_seq = len(y_meas)
        theta = params["theta"]
        theta_rep = jnp.repeat(theta[None], self.n_seq, axis=0)
        obs_theta = jnp.hstack((y_meas, theta_rep))
        return obs_theta

    def _par_parse(self, params, obs_theta):
        mean_model = params["mean"]
        lower_model = params["lower"]
        diag_model = params["diag"]
        mean = jax.vmap(mean_model)(obs_theta)
        lower_chol = jax.vmap(lower_model)(jnp.concatenate([obs_theta[:-1], obs_theta[1:]], axis=1)).reshape((-1, self.n_state, self.n_state)) 
        diag_chol = jnp.zeros((self.n_state, self.n_state, self.n_seq))
        upper_ind = jnp.triu_indices(self.n_state)
        diag_out = jax.vmap(diag_model)(obs_theta)
        diag_chol = diag_chol.at[upper_ind].set(diag_out.T) 
        diag_ind = jnp.diag_indices(self.n_state)
        diag_chol = diag_chol.at[diag_ind].set(jnp.abs(diag_chol[diag_ind])).T + jnp.eye(self.n_state)
        return mean, lower_chol, diag_chol

    def simulate(self, key, params, y_meas, n_sim):
        obs_theta = self._theta_y_meas(params, y_meas)
        mean, lower_chol, diag_chol = self._par_parse(params, obs_theta)
        entropy = btp_entropy(diag_chol)
        return jnp.repeat(mean[None], n_sim, axis=0) + btp_simulate(key, lower_chol, diag_chol, n_sim).transpose(2,0,1), entropy*n_sim

    def post_mv(self, params, y_meas):
        obs_theta = self._theta_y_meas(params, y_meas)
        mean, lower_chol, diag_chol = self._par_parse(params, obs_theta)
        var = btp_var(lower_chol, diag_chol)
        return mean, var

# Smooth model
class SmoothModel:
    r"""
    Uses an RNN as the recognition model, where the output is a jax array of parameters.
    The parameters are split into: {mean/var}_state_filt, {mean/var/wgt}_state.
    
    The underlying state-space model is of the form:

        X_n = Q_n X_{n-1} + c_n + R_n^{1/2} \eps_n

    where Q_n is wgt_state, c_n is mean_state, and R_n is var_state.
    These values are then used to compute ``(mu_{n+1|n}, Sigma_{n+1|n})``.
    Finally, simulation is done using Kalman recursions.
    """
    def __init__(self, n_state):
        self.n_state = n_state

    def _theta_y_meas(self, params, y_meas):
        self.n_seq = len(y_meas)
        theta = params["theta"]
        theta_rep = jnp.repeat(theta[None], self.n_seq, axis=0)
        obs_theta = jnp.hstack((y_meas, theta_rep))
        return obs_theta
    
    def _par_parse(self, params, obs_theta):
        gru_model = params["gru"]
        full_par = gru_model(obs_theta)
        # split parameters into mean_state_filt, mean_state, wgt_state, var_state_filt, var_state
        par_indices = [self.n_state, 2*self.n_state, self.n_state*(2+self.n_state), self.n_state*(3*self.n_state+5)//2, self.n_state*(2*self.n_state+3)]
        mean_state_filt = full_par[:, :par_indices[0]]
        mean_state = full_par[:, par_indices[0]:par_indices[1]]
        wgt_state = full_par[:, par_indices[1]:par_indices[2]].reshape(self.n_seq, self.n_state, self.n_state)
        upper_ind = jnp.triu_indices(self.n_state)
        chol_state_filt = jnp.zeros((self.n_state, self.n_state, self.n_seq))
        chol_state_filt = chol_state_filt.at[upper_ind].set(full_par[:, par_indices[2]:par_indices[3]].T).T*0.1
        chol_state = jnp.zeros((self.n_state, self.n_state, self.n_seq))
        chol_state = chol_state.at[upper_ind].set(full_par[:, par_indices[3]:par_indices[4]].T).T
        # convert cholesky to variance
        def chol_to_var(chol_mat):
            var_mat = chol_mat.dot(chol_mat.T)
            return var_mat
        var_state_filt = jax.vmap(chol_to_var)(chol_state_filt)
        var_state = jax.vmap(chol_to_var)(chol_state)

        # compute predicted values
        mean_state_pred, var_state_pred = jax.vmap(rodeo.kalmantv.predict)(
            mean_state_past=mean_state_filt,
            var_state_past=var_state_filt,
            mean_state=mean_state,
            wgt_state=wgt_state,
            var_state=var_state
        )
        return mean_state_filt, var_state_filt, mean_state_pred, var_state_pred, wgt_state
    
    def simulate(self, key, params, y_meas, n_sim):
        obs_theta = self._theta_y_meas(params, y_meas)
        mean_state_filt, var_state_filt, \
            mean_state_pred, var_state_pred, \
                wgt_state = self._par_parse(params, obs_theta)
        
        # simulate using kalman sampler
        def scan_fun(carry, smooth_kwargs):
            mean_state_filt = smooth_kwargs['mean_state_filt']
            var_state_filt = smooth_kwargs['var_state_filt']
            mean_state_pred = smooth_kwargs['mean_state_pred']
            var_state_pred = smooth_kwargs['var_state_pred']
            wgt_state = smooth_kwargs['wgt_state']
            random_normal = smooth_kwargs['random_normal']
            x_state_next = carry['x_state_next']
            half_det = carry['half_det']
            mean_state_sim, var_state_sim = smooth_sim(
                x_state_next=x_state_next,
                wgt_state=wgt_state,
                mean_state_filt=mean_state_filt,
                var_state_filt=var_state_filt,
                mean_state_pred=mean_state_pred,
                var_state_pred=var_state_pred
            )
            chol_factor = jnp.linalg.cholesky(var_state_sim)
            x_state_curr = mean_state_sim + chol_factor.dot(random_normal).transpose(1,0)
            half_det += jnp.sum(jnp.log(jnp.diag(chol_factor)))
            carry = {
                'x_state_next': x_state_curr,
                'var_state_sim': var_state_sim,
                'half_det': half_det
            }
            return carry, carry
        
        # time N
        random_normals = jax.random.normal(key, shape=(self.n_seq, self.n_state, n_sim))
        chol_factor = jnp.linalg.cholesky(var_state_filt[self.n_seq-1])
        x_init = jnp.repeat(mean_state_filt[self.n_seq-1][None], n_sim, axis=0) + chol_factor.dot(random_normals[self.n_seq-1]).transpose(1,0)
        half_det_init = jnp.sum(jnp.log(jnp.diag(chol_factor)))
        scan_init = {
            'x_state_next': x_init,
            'var_state_sim': var_state_filt[self.n_seq-1],
            'half_det': half_det_init
        }
        # scan arguments
        scan_kwargs = {
            'mean_state_filt': mean_state_filt[:self.n_seq-1],
            'var_state_filt': var_state_filt[:self.n_seq-1],
            'mean_state_pred': mean_state_pred[:self.n_seq-1],
            'var_state_pred': var_state_pred[:self.n_seq-1],
            'wgt_state': wgt_state[:self.n_seq-1],
            'random_normal': random_normals[:self.n_seq-1]
        }

        last_out, stack_out = jax.lax.scan(scan_fun, scan_init, scan_kwargs, reverse=True)
        x_state_smooth = jnp.concatenate(
            [stack_out['x_state_next'], x_init[None]]
        )
        x_state_smooth = x_state_smooth.transpose(1,0,2)
        half_det = last_out["half_det"] # do not change class members
        entropy = half_det + self.n_seq*self.n_state/2.0*(1+jnp.log(2*jnp.pi))
        entropy = entropy*n_sim
        return x_state_smooth, entropy
    
    def post_mv(self, params, y_meas):
        obs_theta = self._theta_y_meas(params, y_meas)
        mean_state_filt, var_state_filt, \
            mean_state_pred, var_state_pred, \
                wgt_state = self._par_parse(params, obs_theta)
        
        # compute kalman smoother
        def scan_fun(carry, smooth_kwargs):
            mean_state_filt = smooth_kwargs['mean_state_filt']
            var_state_filt = smooth_kwargs['var_state_filt']
            mean_state_pred = smooth_kwargs['mean_state_pred']
            var_state_pred = smooth_kwargs['var_state_pred']
            wgt_state = smooth_kwargs['wgt_state']
            mean_state_next = carry['mean_state_next']
            var_state_next = carry['var_state_next']
            mean_state_smooth, var_state_smooth = rodeo.kalmantv.smooth_mv(
                mean_state_next=mean_state_next,
                var_state_next=var_state_next,
                mean_state_filt=mean_state_filt,
                var_state_filt=var_state_filt,
                mean_state_pred=mean_state_pred,
                var_state_pred=var_state_pred,
                wgt_state=wgt_state
            )
            carry = {
                'mean_state_next': mean_state_smooth,
                'var_state_next': var_state_smooth
            }
            return carry, carry
        
        scan_init = {
            'mean_state_next': mean_state_filt[self.n_seq-1],
            'var_state_next': var_state_filt[self.n_seq-1]
        }
        # scan arguments
        scan_kwargs = {
            'mean_state_filt': mean_state_filt[:self.n_seq-1],
            'var_state_filt': var_state_filt[:self.n_seq-1],
            'mean_state_pred': mean_state_pred[:self.n_seq-1],
            'var_state_pred': var_state_pred[:self.n_seq-1],
            'wgt_state': wgt_state[:self.n_seq-1]
        }

        last_out, stack_out = jax.lax.scan(scan_fun, scan_init, scan_kwargs, reverse=True)
        mean_state_smooth = jnp.concatenate(
            [stack_out['mean_state_next'], mean_state_filt[self.n_seq-1][None]]
        )
        var_state_smooth = jnp.concatenate(
            [stack_out['var_state_next'], var_state_filt[self.n_seq-1][None]]
        )
        return mean_state_smooth, var_state_smooth
    

class BiRNNModel:
    r"""
    Uses a Bi-RNN as the recognition model, where the output is a jax array of parameters.
    The parameters are split into: {mean/var}_state_smooth. That is the posterior is assumed
    to be a mean-field Gaussian and simulation is done via N(mu, Sigma) where
    mu := mean_state_smooth and Sigma := var_state_smooth.
    """

    def __init__(self, n_state):
        self.n_state = n_state

    def _theta_y_meas(self, params, y_meas):
        self.n_seq = len(y_meas)
        theta = params["theta"]
        theta_rep = jnp.repeat(theta[None], self.n_seq, axis=0)
        obs_theta = jnp.hstack((y_meas, theta_rep))
        return obs_theta
    
    def _par_parse(self, params, obs_theta):
        bigru_model = params["bigru"]
        full_par = bigru_model(obs_theta)
        # split parameters into mean_state_smoothed, var_state_smoothed
        par_indices = [self.n_state, self.n_state*(3+self.n_state)//2]
        upper_ind = jnp.triu_indices(self.n_state)
        mean_state_smooth = full_par[:, :par_indices[0]]
        chol_state_smooth = jnp.zeros((self.n_state, self.n_state, self.n_seq))
        chol_state_smooth = chol_state_smooth.at[upper_ind].set(full_par[:, par_indices[0]:par_indices[1]].T).T
        # convert cholesky to variance
        def chol_to_var(chol_mat):
            var_mat = chol_mat.dot(chol_mat.T)
            return var_mat
        var_state_smooth = jax.vmap(chol_to_var)(chol_state_smooth)

        return mean_state_smooth, var_state_smooth
    
    def simulate(self, key, params, y_meas, n_sim):
        obs_theta = self._theta_y_meas(params, y_meas)
        mean_state_smooth, var_state_smooth = self._par_parse(params, obs_theta)
        # simulate assuming mean-field Gaussian
        def scan_fun(carry, smooth_kwargs):
            mean_state_smooth = smooth_kwargs['mean_state_smooth']
            var_state_smooth = smooth_kwargs['var_state_smooth']
            random_normal = smooth_kwargs['random_normal']
            half_det = carry['half_det']
            chol_factor = jnp.linalg.cholesky(var_state_smooth)
            x_state_curr = jnp.repeat(mean_state_smooth[None], n_sim, axis=0) + chol_factor.dot(random_normal).transpose(1,0)
            half_det += jnp.sum(jnp.log(jnp.diag(chol_factor)))
            carry = {
                'x_state_next': x_state_curr,
                'half_det': half_det
            }
            return carry, carry
        
        random_normals = jax.random.normal(key, shape=(self.n_seq, self.n_state, n_sim))
        scan_init = {
            'x_state_next': jnp.zeros((n_sim, self.n_state)),
            'half_det': 0.0
        }
        # scan arguments
        scan_kwargs = {
            'mean_state_smooth': mean_state_smooth,
            'var_state_smooth': var_state_smooth,
            'random_normal': random_normals
        }

        last_out, stack_out = jax.lax.scan(scan_fun, scan_init, scan_kwargs)
        x_state_smooth = stack_out['x_state_next'].transpose(1,0,2)
        half_det = last_out["half_det"] # do not change class members
        entropy = half_det + self.n_seq*self.n_state/2.0*(1+jnp.log(2*jnp.pi))
        entropy = entropy*n_sim
        return x_state_smooth, entropy
    
    def post_mv(self, params, y_meas):
        obs_theta = self._theta_y_meas(params, y_meas)
        mean_state_filt, var_state_filt = self._par_parse(params, obs_theta)
        return mean_state_filt, var_state_filt

class SmoothMFModel:
    r"""
    Uses an RNN as the recognition model, where the output is a jax array of parameters.
    The parameters are split into: {mean/var}_state_filt, {mean/var/wgt}_state.
    
    The underlying state-space model is of the form:

        X_n = Q_n X_{n-1} + c_n + R_n^{1/2} \eps_n

    where Q_n is wgt_state, c_n is mean_state, and R_n is var_state.
    These values are then used to compute ``(mu_{n+1|n}, Sigma_{n+1|n})``.
    Finally, simulation is done using Kalman recursions assuming a mean-field VI.
    """
    def __init__(self, n_state):
        self.n_state = n_state

    def _theta_y_meas(self, params, y_meas):
        self.n_seq = len(y_meas)
        theta = params["theta"]
        theta_rep = jnp.repeat(theta[None], self.n_seq, axis=0)
        obs_theta = jnp.hstack((y_meas, theta_rep))
        return obs_theta
    
    def _par_parse(self, params, obs_theta):
        gru_model = params["gru"]
        full_par = gru_model(obs_theta)
        # split parameters into mean_state_filt, mean_state, wgt_state, var_state_filt, var_state
        par_indices = [self.n_state, 2*self.n_state, self.n_state*(2+self.n_state), self.n_state*(3*self.n_state+5)//2, self.n_state*(2*self.n_state+3)]
        mean_state_filt = full_par[:, :par_indices[0]]
        mean_state = full_par[:, par_indices[0]:par_indices[1]]
        wgt_state = full_par[:, par_indices[1]:par_indices[2]].reshape(self.n_seq, self.n_state, self.n_state)
        upper_ind = jnp.triu_indices(self.n_state)
        chol_state_filt = jnp.zeros((self.n_state, self.n_state, self.n_seq))
        chol_state_filt = chol_state_filt.at[upper_ind].set(full_par[:, par_indices[2]:par_indices[3]].T).T*0.1
        chol_state = jnp.zeros((self.n_state, self.n_state, self.n_seq))
        chol_state = chol_state.at[upper_ind].set(full_par[:, par_indices[3]:par_indices[4]].T).T
        # convert cholesky to variance
        def chol_to_var(chol_mat):
            var_mat = chol_mat.dot(chol_mat.T)
            return var_mat
        var_state_filt = jax.vmap(chol_to_var)(chol_state_filt)
        var_state = jax.vmap(chol_to_var)(chol_state)

        # compute predicted values
        mean_state_pred, var_state_pred = jax.vmap(rodeo.kalmantv.predict)(
            mean_state_past=mean_state_filt,
            var_state_past=var_state_filt,
            mean_state=mean_state,
            wgt_state=wgt_state,
            var_state=var_state
        )
        return mean_state_filt, var_state_filt, mean_state_pred, var_state_pred, wgt_state
    
    def simulate(self, key, params, y_meas, n_sim):
        obs_theta = self._theta_y_meas(params, y_meas)
        mean_state_filt, var_state_filt, \
            mean_state_pred, var_state_pred, \
                wgt_state = self._par_parse(params, obs_theta)
        
        # simulate using kalman sampler
        def scan_fun(carry, smooth_kwargs):
            mean_state_filt = smooth_kwargs['mean_state_filt']
            var_state_filt = smooth_kwargs['var_state_filt']
            mean_state_pred = smooth_kwargs['mean_state_pred']
            var_state_pred = smooth_kwargs['var_state_pred']
            wgt_state = smooth_kwargs['wgt_state']
            random_normal = smooth_kwargs['random_normal']
            mean_state_next = carry['mean_state_smooth']
            var_state_next = carry['var_state_smooth']
            half_det = carry['half_det']
            mean_state_smooth, var_state_smooth = rodeo.kalmantv.smooth_mv(
                mean_state_next=mean_state_next,
                var_state_next=var_state_next,
                wgt_state=wgt_state,
                mean_state_filt=mean_state_filt,
                var_state_filt=var_state_filt,
                mean_state_pred=mean_state_pred,
                var_state_pred=var_state_pred
            )
            chol_factor = jnp.linalg.cholesky(var_state_smooth)
            x_state_curr = jnp.repeat(mean_state_smooth[None], n_sim, axis=0) + chol_factor.dot(random_normal).transpose(1,0)
            half_det += jnp.sum(jnp.log(jnp.diag(chol_factor)))
            carry = {
                'mean_state_smooth': mean_state_smooth,
                'var_state_smooth': var_state_smooth,
                'x_state_smooth': x_state_curr,
                'half_det': half_det
            }
            return carry, carry
        
        # time N
        random_normals = jax.random.normal(key, shape=(self.n_seq, self.n_state, n_sim))
        chol_factor = jnp.linalg.cholesky(var_state_filt[self.n_seq-1])
        x_init = jnp.repeat(mean_state_filt[self.n_seq-1][None], n_sim, axis=0) + chol_factor.dot(random_normals[self.n_seq-1]).transpose(1,0)
        half_det_init = jnp.sum(jnp.log(jnp.diag(chol_factor)))
        scan_init = {
            'mean_state_smooth': mean_state_filt[self.n_seq-1],
            'var_state_smooth': var_state_filt[self.n_seq-1],
            'x_state_smooth': x_init,
            'half_det': half_det_init
        }
        # scan arguments
        scan_kwargs = {
            'mean_state_filt': mean_state_filt[:self.n_seq-1],
            'var_state_filt': var_state_filt[:self.n_seq-1],
            'mean_state_pred': mean_state_pred[:self.n_seq-1],
            'var_state_pred': var_state_pred[:self.n_seq-1],
            'wgt_state': wgt_state[:self.n_seq-1],
            'random_normal': random_normals[:self.n_seq-1]
        }

        last_out, stack_out = jax.lax.scan(scan_fun, scan_init, scan_kwargs, reverse=True)
        x_state_smooth = jnp.concatenate(
            [stack_out['x_state_smooth'], x_init[None]]
        )
        x_state_smooth = x_state_smooth.transpose(1,0,2)
        half_det = last_out["half_det"] # do not change class members
        entropy = half_det + self.n_seq*self.n_state/2.0*(1+jnp.log(2*jnp.pi))
        entropy = entropy*n_sim
        return x_state_smooth, entropy
    
    def post_mv(self, params, y_meas):
        obs_theta = self._theta_y_meas(params, y_meas)
        mean_state_filt, var_state_filt, \
            mean_state_pred, var_state_pred, \
                wgt_state = self._par_parse(params, obs_theta)
        
        # compute kalman smoother
        def scan_fun(carry, smooth_kwargs):
            mean_state_filt = smooth_kwargs['mean_state_filt']
            var_state_filt = smooth_kwargs['var_state_filt']
            mean_state_pred = smooth_kwargs['mean_state_pred']
            var_state_pred = smooth_kwargs['var_state_pred']
            wgt_state = smooth_kwargs['wgt_state']
            mean_state_next = carry['mean_state_next']
            var_state_next = carry['var_state_next']
            mean_state_smooth, var_state_smooth = rodeo.kalmantv.smooth_mv(
                mean_state_next=mean_state_next,
                var_state_next=var_state_next,
                mean_state_filt=mean_state_filt,
                var_state_filt=var_state_filt,
                mean_state_pred=mean_state_pred,
                var_state_pred=var_state_pred,
                wgt_state=wgt_state
            )
            carry = {
                'mean_state_next': mean_state_smooth,
                'var_state_next': var_state_smooth
            }
            return carry, carry
        
        scan_init = {
            'mean_state_next': mean_state_filt[self.n_seq-1],
            'var_state_next': var_state_filt[self.n_seq-1]
        }
        # scan arguments
        scan_kwargs = {
            'mean_state_filt': mean_state_filt[:self.n_seq-1],
            'var_state_filt': var_state_filt[:self.n_seq-1],
            'mean_state_pred': mean_state_pred[:self.n_seq-1],
            'var_state_pred': var_state_pred[:self.n_seq-1],
            'wgt_state': wgt_state[:self.n_seq-1]
        }

        last_out, stack_out = jax.lax.scan(scan_fun, scan_init, scan_kwargs, reverse=True)
        mean_state_smooth = jnp.concatenate(
            [stack_out['mean_state_next'], mean_state_filt[self.n_seq-1][None]]
        )
        var_state_smooth = jnp.concatenate(
            [stack_out['var_state_next'], var_state_filt[self.n_seq-1][None]]
        )
        return mean_state_smooth, var_state_smooth
    