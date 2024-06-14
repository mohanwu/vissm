r"""
This module implement the RNN method of Ryder at al. (2018)

RNN takes on the features:
- $\theta$
- $\x_n$
- $t'_m - t_n$
- $t'_m$
- $y_m$

where $t_0, ... t_N$ is the SDE time grid and $t'_0, ..., t'_M$ is the observation time grid.
Here $t'_m$ is the next observation time greater than $t_n$. 

The RNN returns $\alpha, M$ where simulation is done via $x_n = x_{n-1} + \alpha \delta t + M \sqrt \delta t \epsilon$.  

"""
import jax
import jax.numpy as jnp
import equinox as eqx

def theta_to_chol(theta_lower, n_theta):
    lower_ind = jnp.tril_indices(n_theta)
    theta_chol = jnp.zeros((n_theta, n_theta))
    theta_chol = theta_chol.at[lower_ind].set(theta_lower)
    return theta_chol

# Smooth model
class RyderRNN(eqx.Module):
    hidden_size: int
    out_size: int
    layers: list
    linear: eqx.Module
     
    def __init__(self, key, n_state, n_inp):
        key, *subkey = jax.random.split(key, num=6)
        self.out_size = n_state + n_state*(n_state + 1)//2
        self.hidden_size = 50
        self.layers = [
            eqx.nn.GRUCell(n_inp, self.hidden_size, key=subkey[0]),
            eqx.nn.GRUCell(self.hidden_size, self.hidden_size, key=subkey[1]),
            eqx.nn.GRUCell(self.hidden_size, self.hidden_size, key=subkey[2]),
            eqx.nn.GRUCell(self.hidden_size, self.hidden_size, key=subkey[3]),
        ]
        self.linear = eqx.nn.Linear(self.hidden_size, self.out_size, key=subkey[4])
    

    # GRU(y_t,h_t) -> h_{t+1}
    def __call__(self, input):
        hidden = jnp.zeros((4, self.hidden_size,))
        # hidden2 = jnp.zeros((self.hidden_size,))
        data_seq = input
        for i in range(len(hidden)):
            def f(carry, inp):
                return self.layers[i](inp, carry), self.layers[i](inp, carry)
            final, data_seq = jax.lax.scan(f, hidden[i], data_seq)
        out = jax.vmap(self.linear)(data_seq)
        return out


class Ryder:

    def __init__(self, n_state, obs_times, sde_times, x_init):
        self._n_state = n_state
        self._obs_times = obs_times
        self._sde_times = sde_times
        self._dt = sde_times[1] - sde_times[0]
        self._x_init = x_init
    
    def _rnn_input(self, theta, y_meas):
        theta_rep = jnp.repeat(theta[None], len(self._sde_times)-1, axis=0)
        obs_ind = jnp.searchsorted(self._obs_times, self._sde_times[:-1], side='right')
        time_next = self._obs_times[obs_ind]
        time_diff = time_next - self._sde_times[:-1]
        y_meas_comb = jnp.hstack([y_meas[:-1], y_meas[1:]])
        y_meas_prev_next = y_meas_comb[obs_ind-1]
        # y_meas_prev_next = y_meas[obs_ind]
        input = jnp.concatenate([theta_rep, self._sde_times[:-1, None], time_diff[:, None], y_meas_prev_next], axis=1)
        return input
    
    def _par_parse(self, params, rnn_input):
        gru_model = params["gru"]
        full_par = gru_model(rnn_input)
        # split parameters into alpha, beta
        alpha = full_par[:, :self._n_state]
        upper_ind = jnp.triu_indices(self._n_state)
        beta_lower = jnp.zeros((self._n_state, self._n_state, len(full_par)))
        beta_lower = beta_lower.at[upper_ind].set(full_par[:, self._n_state:].T).T
        # beta_lower = beta_lower.at[jnp.diag_indices(self._n_state)].set(jax.nn.softplus(beta_lower[jnp.diag_indices(self._n_state)])).T + \
        #             1e-3*jnp.eye(self._n_state)
        return alpha, beta_lower
    
    def simulate(self, key, params, y_meas):
        # generate theta
        key, subkey = jax.random.split(key)
        theta_mu = params["theta_mu"]
        n_theta = len(theta_mu)

        # theta_chol = theta_to_chol(params["theta_chol"], n_theta)
        # theta_chol = theta_chol.at[jnp.diag_indices(n_theta)].set(jax.nn.softplus(jnp.diag(theta_chol)))
        theta_std = jax.nn.softplus(params["theta_std"])
        random_normal = jax.random.normal(subkey, shape=(n_theta,))
        theta = theta_mu + theta_std*random_normal
        # theta = jnp.exp(theta)
        # theta = theta_mu + theta_chol.dot(random_normal)
        rnn_input = self._rnn_input(theta, y_meas)
        alpha, beta_lower = self._par_parse(params, rnn_input)

        def scan_fun(carry, args):
            alpha = args['alpha']
            beta_lower = args['beta_lower']
            random_normal = args['random_normal']
            x_prev = carry['x_curr']
            x_neglogpdf = carry["x_neglogpdf"]
            beta = beta_lower.dot(beta_lower.T)
            beta_lower = jnp.linalg.cholesky(beta)
            x_curr = jax.nn.softplus(x_prev + alpha * self._dt + beta_lower.dot(random_normal) * jnp.sqrt(self._dt))
            x_neglogpdf -= jax.scipy.stats.multivariate_normal.logpdf(x_curr, x_prev + alpha * self._dt, beta * self._dt)
            carry = {
                'x_curr': x_curr,
                'x_neglogpdf': x_neglogpdf
            }
            return carry, carry
        
        # time 0
        x0 = self._x_init
        scan_init = {
            'x_curr': x0,
            'x_neglogpdf': 0.0
        }
        sde_len = len(self._sde_times)
        random_normals = jax.random.normal(key, shape=(sde_len-1, self._n_state))
        # scan arguments
        scan_kwargs = {
            'alpha': alpha,
            'beta_lower': beta_lower,
            'random_normal': random_normals
        }

        last_out, stack_out = jax.lax.scan(scan_fun, scan_init, scan_kwargs)
        xs = jnp.concatenate(
            [x0[None], stack_out['x_curr']]
        )
        # calculate -E[log q(x, theta|phi_full)]
        # use entropy for - E[log q(theta)]
        # use negative logpdf for - E[log q(x|theta)]
        x_neglogpdf = last_out["x_neglogpdf"]
        # theta_entpy = 0.5*n_theta*(1+jnp.log(2*jnp.pi)) + jnp.sum(jnp.log(jnp.diag(theta_chol)))
        theta_entpy = 0.5*n_theta*(1+jnp.log(2*jnp.pi)) + jnp.sum(jnp.log(theta_std))
        theta_x_neglogpdf = x_neglogpdf + theta_entpy
        return (xs, theta), theta_x_neglogpdf


        # rnn
