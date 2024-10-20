import jax
import jax.numpy as jnp
import rodeo.kalmantv
import equinox as eqx


def theta_to_chol(theta_lower, n_theta):
    lower_ind = jnp.tril_indices(n_theta)
    diag_ind = jnp.diag_indices(n_theta)
    theta_chol = jnp.zeros((n_theta, n_theta))
    theta_chol = theta_chol.at[lower_ind].set(theta_lower)
    theta_chol = theta_chol.at[diag_ind].set(jax.nn.softplus(jnp.diag(theta_chol)))
    return theta_chol


# NN for random effects
class NN_rand(eqx.Module):
    hidden_size: int
    out_size: int
    layers: list
    linear: eqx.Module
     
    def __init__(self, key, n_theta, n_obs, n_effect):
        key, *subkey = jax.random.split(key, num=5)
        self.out_size = n_effect * n_theta + n_effect + n_effect * (n_effect + 1) // 2
        n_inp = n_obs + n_theta
        self.hidden_size = 20
        self.layers = [
            eqx.nn.Linear(n_inp, 2*n_inp, key=subkey[0]),
            jax.nn.relu,
            eqx.nn.Linear(2*n_inp, 4*n_inp, key=subkey[1]),
            jax.nn.relu,
            eqx.nn.Linear(4*n_inp, 2*n_inp, key=subkey[2]),
            jax.nn.relu,
            # eqx.nn.Linear(2*n_inp, n_inp, key=subkey[3]),
            # jax.nn.relu,
            # eqx.nn.Linear(n_inp, n_inp, key=subkey[4]),
            # jax.nn.relu,

        ]
        self.linear = eqx.nn.Linear(2*n_inp, self.out_size, key=subkey[3])

    def __call__(self, input):
        
        for layer in self.layers:
            input = layer(input)
        
        out = self.linear(input)
        return out

class RNN_rand(eqx.Module):
    hidden_size: int
    layers: list
    linear: eqx.Module
     
    def __init__(self, key, n_theta, n_obs, n_effect):
        key, *subkey = jax.random.split(key, num=7)
        n_inp = n_theta + n_obs
        self.hidden_size = n_inp
        self.layers = [
            eqx.nn.GRUCell(self.hidden_size, self.hidden_size, key=subkey[0]),
            eqx.nn.GRUCell(self.hidden_size, self.hidden_size, key=subkey[1]),
            # eqx.nn.GRUCell(self.hidden_size, self.hidden_size, key=subkey[2]),
            # eqx.nn.GRUCell(self.hidden_size, self.hidden_size, key=subkey[3]),
        ]
        self.linear = eqx.nn.Linear(self.hidden_size, n_effect, key=subkey[2])
    

    # GRU(y_t,h_t) -> h_{t+1}
    def __call__(self, y_meas):
        hidden = jnp.zeros((len(self.layers), self.hidden_size,))
        data_seq = y_meas
        for i in range(len(hidden)):
            def f(carry, inp):
                return self.layers[i](inp, carry), self.layers[i](inp, carry)
            final, data_seq = jax.lax.scan(f, hidden[i], data_seq)
        out = self.linear(final)
        return out


class RandomModel:
    
    def __init__(self, n_state, random_ind, fixed_ind, obs_times, sde_times, x_init):
        self._n_state = n_state
        # self._random_mu_ind = random_mu_ind
        # self._random_std_ind = random_std_ind
        self._random_ind = random_ind
        self._n_random = len(random_ind)//2
        self._fixed_ind = fixed_ind
        self._obs_times = obs_times
        self._sde_times = sde_times
        self._dt = sde_times[1] - sde_times[0]
        self._x_init = x_init
        self._n_sde = len(sde_times)

    def _sim_random(self, key, params, theta, y_meas):
        n_sim, n_obs = y_meas.shape[0:2]        
        random_normals = jax.random.normal(key, shape=(n_sim, self._n_random))
        n_theta = len(theta) 
        # idx_mu = jnp.array([0,2])
        idx = jnp.array([1,3,4])
        theta = theta.at[idx].set(jnp.exp(theta[idx]))
        # theta = jnp.repeat(theta[None], n_obs, axis=0)
        model = params["nn_random"]
        # model2 = params["nn_random2"]
        def vmap_fun(random_normal, y_n):
            y_theta = jnp.append(y_n, theta)
            model_output = model(y_theta)
            wgt_ind = self._n_random * n_theta
            wgt_theta = model_output[:wgt_ind].reshape(self._n_random, n_theta)*0.01 + jnp.eye(self._n_random, n_theta)
            mu_theta = model_output[wgt_ind:wgt_ind+self._n_random]
            lower_theta = model_output[wgt_ind+self._n_random:]
            chol_theta = theta_to_chol(lower_theta, self._n_random)
            random_mu = wgt_theta.dot(theta) + mu_theta
            # random_mu = theta
            # random_std = jax.nn.softplus(model_output[self._n_random:])
            random_effect = random_mu + chol_theta.dot(random_normal)
            nlp = -jnp.sum(jax.scipy.stats.multivariate_normal.logpdf(random_effect, random_mu, chol_theta.dot(chol_theta.T)))
            return random_effect, nlp
        
        random_effect, nlp = jax.vmap(vmap_fun)(random_normals, y_meas)
        return random_effect, jnp.sum(nlp)


    def simulate(self, key, params, y_meas):
        key, *subkeys = jax.random.split(key, num=4)
        # simulate theta
        theta_mu = params["theta_mu"]
        n_theta = len(theta_mu)
        
        # theta_std = jax.nn.softplus(params["theta_std"])
        theta_normal = jax.random.normal(subkeys[0], shape=(n_theta,))
        theta_chol = theta_to_chol(params["theta_chol"], n_theta)
        theta = theta_mu + theta_chol.dot(theta_normal)
        # theta = theta_mu + theta_std*theta_normal
        # simulate random effect
        # nn_random_model = params["nn_random1"]
        idx = jnp.array([0,2])
        random_effect, random_neglogpdf = self._sim_random(subkeys[1], params, theta, y_meas)
        # calculate -E[log q(theta|phi_full)]
        # use entropy for - E[log q(theta)]
        # use negative logpdf for - E[log q(eta|theta)]
        # theta_entpy = 0.5*n_theta*(1+jnp.log(2*jnp.pi)) + jnp.sum(jnp.log(theta_std))
        # theta_entpy = -jnp.sum(jax.scipy.stats.norm.logpdf(theta, theta_mu, theta_std))
        theta_entpy = -jnp.sum(jax.scipy.stats.multivariate_normal.logpdf(theta, theta_mu, theta_chol.dot(theta_chol.T)))
        # theta_entpy = 0.5*n_theta*(1+jnp.log(2*jnp.pi)) + jnp.sum(jnp.log(jnp.diag(theta_chol)))
        nlp = theta_entpy + random_neglogpdf
        return (theta, random_effect), nlp
