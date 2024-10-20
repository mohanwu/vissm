import jax
import jax.numpy as jnp
import rodeo.kalmantv
import equinox as eqx
from vissm.models import NN_Mean, NN_Lower, NN_Diag, RNN, BIRNN
from vissm.block_tridiag import *

def theta_to_chol(theta_lower, n_theta):
    lower_ind = jnp.tril_indices(n_theta)
    theta_chol = jnp.zeros((n_theta, n_theta))
    theta_chol = theta_chol.at[lower_ind].set(theta_lower)
    return theta_chol

# # archer model

class ArcherModel:
    def __init__(self, n_state):
        self.n_state = n_state
    
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

    def simulate(self, key, params, y_meas):
        # draw from theta ~ N(mu, chol)
        key, subkey = jax.random.split(key)
        self.n_seq = len(y_meas)
        theta_mu = params["theta_mu"]
        n_theta = len(theta_mu)
        theta_chol = theta_to_chol(params["theta_chol"], n_theta)
        random_normal = jax.random.normal(subkey, shape=(n_theta,))
        theta = theta_mu + theta_chol.dot(random_normal)
        theta_rep = jnp.repeat(theta[None], self.n_seq, axis=0)
        obs_theta = jnp.hstack((y_meas, theta_rep))
        mean, lower_chol, diag_chol = self._par_parse(params, obs_theta)
        x = mean + btp_simulate(key, lower_chol, diag_chol, n_sim=1)[:, :, 0]
        # calculate -E[log q(x, theta|phi_full)]
        # use entropy for - E[log q(theta)]
        # use negative logpdf for - E[log q(x|theta)]
        x_neglogpdf = -btp_logpdf(x-mean, lower_chol, diag_chol)
        theta_entpy = 0.5*n_theta*(1+jnp.log(2*jnp.pi)) + jnp.sum(jnp.log(jnp.abs(jnp.diag(theta_chol))))
        theta_x_neglogpdf = x_neglogpdf + theta_entpy
        return (x, theta), theta_x_neglogpdf
    
    def post_mv(self, params, y_meas):
        self.n_seq = len(y_meas)
        theta = params["theta_mu"]
        theta_rep = jnp.repeat(theta[None], self.n_seq, axis=0)
        obs_theta = jnp.hstack((y_meas, theta_rep))
        mean, lower_chol, diag_chol = self._par_parse(params, obs_theta)
        var = btp_var(lower_chol, diag_chol)
        return mean, var

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
    
    def simulate(self, key, params, y_meas):
        key, subkey = jax.random.split(key)
        self.n_seq = len(y_meas)
        theta_mu = params["theta_mu"]
        n_theta = len(theta_mu)
        theta_chol = theta_to_chol(params["theta_chol"], n_theta)
        random_normal = jax.random.normal(subkey, shape=(n_theta,))
        theta = theta_mu + theta_chol.dot(random_normal)
        theta_rep = jnp.repeat(theta[None], self.n_seq, axis=0)
        obs_theta = jnp.hstack((y_meas, theta_rep))
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
            x_neglogpdf = carry["x_neglogpdf"]
            mean_state_sim, var_state_sim = rodeo.kalmantv.smooth_sim(
                x_state_next=x_state_next,
                wgt_state=wgt_state,
                mean_state_filt=mean_state_filt,
                var_state_filt=var_state_filt,
                mean_state_pred=mean_state_pred,
                var_state_pred=var_state_pred
            )
            chol_factor = jnp.linalg.cholesky(var_state_sim)
            x_state_curr = mean_state_sim + chol_factor.dot(random_normal)
            x_neglogpdf -= jax.scipy.stats.multivariate_normal.logpdf(x_state_curr, mean_state_sim, var_state_sim)
            carry = {
                'x_state_next': x_state_curr,
                'x_neglogpdf': x_neglogpdf
            }
            return carry, carry
        
        # time N
        mean_state_N = mean_state_filt[self.n_seq-1]
        var_state_N = var_state_filt[self.n_seq-1]
        random_normals = jax.random.normal(key, shape=(self.n_seq, self.n_state))
        chol_factor = jnp.linalg.cholesky(var_state_N)
        x_N = mean_state_N + chol_factor.dot(random_normals[self.n_seq-1])
        x_neglogpdf = -jax.scipy.stats.multivariate_normal.logpdf(x_N, mean_state_N, var_state_N)
        scan_init = {
            'x_state_next': x_N,
            'x_neglogpdf': x_neglogpdf
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
            [stack_out['x_state_next'], x_N[None]]
        )
        # calculate -E[log q(x, theta|phi_full)]
        # use entropy for - E[log q(theta)]
        # use negative logpdf for - E[log q(x|theta)]
        x_neglogpdf = last_out["x_neglogpdf"]
        theta_entpy = 0.5*n_theta*(1+jnp.log(2*jnp.pi)) + jnp.sum(jnp.log(jnp.abs(jnp.diag(theta_chol))))
        theta_x_neglogpdf = x_neglogpdf + theta_entpy
        return (x_state_smooth, theta), theta_x_neglogpdf
    
    def post_mv(self, params, y_meas):
        self.n_seq = len(y_meas)
        theta = params["theta_mu"]
        theta_rep = jnp.repeat(theta[None], self.n_seq, axis=0)
        obs_theta = jnp.hstack((y_meas, theta_rep))
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
    
    def simulate(self, key, params, y_meas):
        key, subkey = jax.random.split(key)
        self.n_seq = len(y_meas)
        theta_mu = params["theta_mu"]
        n_theta = len(theta_mu)
        theta_chol = theta_to_chol(params["theta_chol"], n_theta)
        random_normal = jax.random.normal(subkey, shape=(n_theta,))
        theta = theta_mu + theta_chol.dot(random_normal)
        theta_rep = jnp.repeat(theta[None], self.n_seq, axis=0)
        obs_theta = jnp.hstack((y_meas, theta_rep))
        mean_state_smooth, var_state_smooth = self._par_parse(params, obs_theta)


        # simulate assuming mean-field Gaussian
        def scan_fun(carry, smooth_kwargs):
            mean_state_smooth = smooth_kwargs['mean_state_smooth']
            var_state_smooth = smooth_kwargs['var_state_smooth']
            random_normal = smooth_kwargs['random_normal']
            x_neglogpdf  = carry['x_neglogpdf']
            chol_factor = jnp.linalg.cholesky(var_state_smooth)
            x_state_curr = mean_state_smooth + chol_factor.dot(random_normal)
            x_neglogpdf -= jax.scipy.stats.multivariate_normal.logpdf(x_state_curr, mean_state_smooth, var_state_smooth)
            carry = {
                'x_state_next': x_state_curr,
                'x_neglogpdf': x_neglogpdf 
            }
            return carry, carry
        
        random_normals = jax.random.normal(key, shape=(self.n_seq, self.n_state))
        scan_init = {
            'x_state_next': jnp.zeros((self.n_state, )),
            'x_neglogpdf': 0.0
        }
        # scan arguments
        scan_kwargs = {
            'mean_state_smooth': mean_state_smooth,
            'var_state_smooth': var_state_smooth,
            'random_normal': random_normals
        }

        last_out, stack_out = jax.lax.scan(scan_fun, scan_init, scan_kwargs)
        x_state_smooth = stack_out['x_state_next']
        x_neglogpdf = last_out["x_neglogpdf"]
        theta_entpy = 0.5*n_theta*(1+jnp.log(2*jnp.pi)) + jnp.sum(jnp.log(jnp.abs(jnp.diag(theta_chol))))
        theta_x_neglogpdf = x_neglogpdf + theta_entpy
        return (x_state_smooth, theta), theta_x_neglogpdf
    
    def post_mv(self, params, y_meas):
        self.n_seq = len(y_meas)
        theta = params["theta_mu"]
        theta_rep = jnp.repeat(theta[None], self.n_seq, axis=0)
        obs_theta = jnp.hstack((y_meas, theta_rep))
        mean_state_smooth, var_state_smooth = self._par_parse(params, obs_theta)
        return mean_state_smooth, var_state_smooth

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
    
    def simulate(self, key, params, y_meas):
        key, subkey = jax.random.split(key)
        self.n_seq = len(y_meas)
        theta_mu = params["theta_mu"]
        n_theta = len(theta_mu)
        theta_chol = theta_to_chol(params["theta_chol"], n_theta)
        random_normal = jax.random.normal(subkey, shape=(n_theta,))
        theta = theta_mu + theta_chol.dot(random_normal)
        theta_rep = jnp.repeat(theta[None], self.n_seq, axis=0)
        obs_theta = jnp.hstack((y_meas, theta_rep))
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
            x_neglogpdf = carry["x_neglogpdf"]
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
            x_state_curr = mean_state_smooth + chol_factor.dot(random_normal)
            x_neglogpdf -= jax.scipy.stats.multivariate_normal.logpdf(x_state_curr, mean_state_smooth, var_state_smooth)
            carry = {
                'mean_state_smooth': mean_state_smooth,
                'var_state_smooth': var_state_smooth,
                'x_state_smooth': x_state_curr,
                'x_neglogpdf': x_neglogpdf
            }
            return carry, carry
        
        # time N
        mean_state_N = mean_state_filt[self.n_seq-1]
        var_state_N = var_state_filt[self.n_seq-1]
        random_normals = jax.random.normal(key, shape=(self.n_seq, self.n_state))
        chol_factor = jnp.linalg.cholesky(var_state_N)
        x_N = mean_state_N + chol_factor.dot(random_normals[self.n_seq-1])
        x_neglogpdf = -jax.scipy.stats.multivariate_normal.logpdf(x_N, mean_state_N, var_state_N)
        scan_init = {
            'mean_state_smooth': mean_state_filt[self.n_seq-1],
            'var_state_smooth': var_state_filt[self.n_seq-1],
            'x_state_smooth': x_N,
            'x_neglogpdf': x_neglogpdf
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
            [stack_out['x_state_smooth'], x_N[None]]
        )
        # calculate -E[log q(x, theta|phi_full)]
        # use entropy for - E[log q(theta)]
        # use negative logpdf for - E[log q(x|theta)]
        x_neglogpdf = last_out["x_neglogpdf"]
        theta_entpy = 0.5*n_theta*(1+jnp.log(2*jnp.pi)) + jnp.sum(jnp.log(jnp.abs(jnp.diag(theta_chol))))
        theta_x_neglogpdf = x_neglogpdf + theta_entpy
        return (x_state_smooth, theta), theta_x_neglogpdf
    
    def post_mv(self, params, y_meas):
        self.n_seq = len(y_meas)
        theta = params["theta_mu"]
        theta_rep = jnp.repeat(theta[None], self.n_seq, axis=0)
        obs_theta = jnp.hstack((y_meas, theta_rep))
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
