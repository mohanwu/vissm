import jax
import jax.numpy as jnp
import rodeo.kalmantv
import equinox as eqx
from vissm.block_tridiag import *
from vissm.utils import smooth_sim

# archer model

class NN_Mean(eqx.Module):
    layers: list

    def __init__(self, key, n_state, n_meas, hidden_size=50):
        key1, key2 = jax.random.split(key)
        self.layers = [
            eqx.nn.Linear(n_meas, hidden_size, key=key1),
            # jnp.tanh,
            jax.nn.hard_tanh,
            eqx.nn.Linear(hidden_size, n_state, key=key2)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class NN_Lower(eqx.Module):
    layers: list

    def __init__(self, key, n_state, n_meas, hidden_size=50):
        key1, key2 = jax.random.split(key)
        self.layers = [
            eqx.nn.Linear(2*n_meas, hidden_size, key=key1),
            jax.nn.hard_tanh,
            eqx.nn.Linear(hidden_size, n_state*n_state, key=key2)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class NN_Diag(eqx.Module):
    layers: list

    def __init__(self, key, n_state, n_meas, hidden_size=50):
        key1, key2 = jax.random.split(key)
        self.layers = [
            eqx.nn.Linear(n_meas, hidden_size, key=key1),
            jax.nn.hard_tanh,
            eqx.nn.Linear(hidden_size, n_state*(n_state+1)//2, key=key2)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ArcherModel:
    def __init__(self, n_state):
        self.n_state = n_state
    
    def _par_parse(self, params, y_meas):
        n_seq = len(y_meas)
        mean_model = params["mean"]
        lower_model = params["lower"]
        diag_model = params["diag"]
        mean = jax.vmap(mean_model)(y_meas)
        lower_chol = jax.vmap(lower_model)(jnp.concatenate([y_meas[:-1], y_meas[1:]], axis=1)).reshape((-1, self.n_state, self.n_state)) 
        diag_chol = jnp.zeros((self.n_state, self.n_state, n_seq))
        upper_ind = jnp.triu_indices(self.n_state)
        diag_out = jax.vmap(diag_model)(y_meas)
        diag_chol = diag_chol.at[upper_ind].set(diag_out.T) 
        diag_ind = jnp.diag_indices(self.n_state)
        diag_chol = diag_chol.at[diag_ind].set(jnp.abs(diag_chol[diag_ind])).T + jnp.eye(self.n_state)
        return mean, lower_chol, diag_chol

    def simulate(self, key, params, y_meas, n_sim):
        mean, lower_chol, diag_chol = self._par_parse(params, y_meas)
        entropy = btp_entropy(diag_chol)
        return jnp.repeat(mean[None], n_sim, axis=0) + btp_simulate(key, lower_chol, diag_chol, n_sim).transpose(2,0,1), entropy*n_sim

    def post_mv(self, params, y_meas):
        mean, lower_chol, diag_chol = self._par_parse(params, y_meas)
        var = btp_var(lower_chol, diag_chol)
        return mean, var

# Smooth model
class RNN(eqx.Module):
    hidden_size: int
    layers: list
    # cell: eqx.Module
    # cell2: eqx.Module
    linear: eqx.Module
     
    def __init__(self, key, n_state, n_meas):
        key, *subkey = jax.random.split(key, num=6)
        self.hidden_size = n_state*(3 + 2*n_state)*2
        self.layers = [
            eqx.nn.GRUCell(n_meas, self.hidden_size, key=subkey[0]),
            eqx.nn.GRUCell(self.hidden_size, self.hidden_size, key=subkey[1]),
            eqx.nn.GRUCell(self.hidden_size, self.hidden_size, key=subkey[2]),
            eqx.nn.GRUCell(self.hidden_size, self.hidden_size, key=subkey[3]),
        ]
        # self.cell = eqx.nn.GRUCell(n_meas, self.hidden_size, key=subkey[0])
        # self.cell2 = eqx.nn.GRUCell(self.hidden_size, self.hidden_size, key=subkey[1])
        # self.cell3 = eqx.nn.GRUCell(self.hidden_size, self.hidden_size, key=subkey[2])
        # self.cell4 = eqx.nn.GRUCell(self.hidden_size, self.hidden_size, key=subkey[3])
        self.linear = eqx.nn.Linear(self.hidden_size, self.hidden_size, key=subkey[4])
    

    # GRU(y_t,h_t) -> h_{t+1}
    def __call__(self, y_meas):
        hidden = jnp.zeros((len(self.layers), self.hidden_size,))
        # hidden2 = jnp.zeros((self.hidden_size,))
        data_seq = y_meas
        for i in range(len(hidden)):
            def f(carry, inp):
                return self.layers[i](inp, carry), self.layers[i](inp, carry)
            final, data_seq = jax.lax.scan(f, hidden[i], data_seq)
        out = jax.vmap(self.linear)(data_seq)
        return out

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

    def _par_parse(self, params, y_meas):
        self.n_seq = len(y_meas)
        gru_model = params["gru"]
        full_par = gru_model(y_meas)
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
        mean_state_filt, var_state_filt, \
            mean_state_pred, var_state_pred, \
                wgt_state = self._par_parse(params, y_meas)
        
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
        mean_state_filt, var_state_filt, \
            mean_state_pred, var_state_pred, \
                wgt_state = self._par_parse(params, y_meas)
        
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
    
class BIRNN(eqx.Module):
    hidden_size: int
    cellf: eqx.Module
    cellb: eqx.Module
    linear: eqx.Module
     
    def __init__(self, key, n_state, n_meas):
        keys = jax.random.split(key, num=3)
        self.hidden_size = n_state*(3 + n_state)
        self.cellf = eqx.nn.GRUCell(n_meas, self.hidden_size//2, key=keys[0])
        self.cellb = eqx.nn.GRUCell(n_meas, self.hidden_size - self.hidden_size//2, key=keys[1])
        self.linear = eqx.nn.Linear(self.hidden_size, self.hidden_size, key=keys[2])

    # GRU(y_t,h_t) -> h_{t+1}
    def __call__(self, y_meas):
        hidden = jnp.zeros((self.hidden_size//2,))
        def forward(carry, inp):
            return self.cellf(inp, carry), self.cellf(inp, carry)
        final, full_f = jax.lax.scan(forward, hidden, y_meas)

        hiddenb = jnp.zeros((self.hidden_size - self.hidden_size//2,))
        def backward(carry, inp):
            return self.cellb(inp, carry), self.cellb(inp, carry)
        final, full_b = jax.lax.scan(backward, hiddenb, y_meas, reverse=True)
        full = jnp.hstack([full_f, full_b])
        out = jax.vmap(self.linear)(full)

        return out

class BiRNNModel:
    r"""
    Uses a Bi-RNN as the recognition model, where the output is a jax array of parameters.
    The parameters are split into: {mean/var}_state_smooth. That is the posterior is assumed
    to be a mean-field Gaussian and simulation is done via N(mu, Sigma) where
    mu := mean_state_smooth and Sigma := var_state_smooth.
    """

    def __init__(self, n_state):
        self.n_state = n_state

    def _par_parse(self, params, y_meas):
        self.n_seq = len(y_meas)
        bigru_model = params["bigru"]
        full_par = bigru_model(y_meas)
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
        mean_state_smooth, var_state_smooth = self._par_parse(params, y_meas)

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
        mean_state_filt, var_state_filt = self._par_parse(params, y_meas)
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

    def _par_parse(self, params, y_meas):
        self.n_seq = len(y_meas)
        gru_model = params["gru"]
        full_par = gru_model(y_meas)
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
        mean_state_filt, var_state_filt, \
            mean_state_pred, var_state_pred, \
                wgt_state = self._par_parse(params, y_meas)
        
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
        mean_state_filt, var_state_filt, \
            mean_state_pred, var_state_pred, \
                wgt_state = self._par_parse(params, y_meas)
        
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
    