import jax
import jax.numpy as jnp
from rodeo.kalmantv import smooth_mv, filter, _smooth


def vec_multivariate_normal_logpdf(x, mean, chol):
    r"""
    Vectorize multivariate normal logpdf, assuming the same Cholesky matrix.

    Args:
        x (ndarray(n, d)): Vector of observations.
        mean (ndarray(n, d)): Vector of means.
        chol (ndarray(d, d)): Cholesky matrix for all observations.

    Returns:
        (float): Total logpdf.

    """
    n, d = x.shape
    res = x-mean
    inv = jax.scipy.linalg.cho_solve((chol, True), jnp.eye(d))

    def inv_each(t):
        return jnp.linalg.multi_dot([res[t].T, inv, res[t]])
    
    inv_prod = jnp.sum(jax.vmap(inv_each)(jnp.arange(n)))*0.5
    const = d*jnp.log(2*jnp.pi)
    det = 2*jnp.sum(jnp.log(jnp.diag(chol)))
    return -0.5*n*(const + det) - inv_prod

# Kalman posterior 
def kalman_post(mean_state_init, var_state_init, 
                mean_state, wgt_state, var_state,
                x_meas, mean_meas, wgt_meas, var_meas):
    r"""
    Function to compute the posterior mean using a linear state-space model with the Kalman filter.
    """
    n_steps = len(x_meas)
    # scan itself
    filt_out = kalman_filter(mean_state_init, var_state_init, 
                  mean_state, wgt_state, var_state,
                  x_meas, mean_meas, wgt_meas, var_meas)

    mean_state_pred, var_state_pred = filt_out["state_pred"]
    mean_state_filt, var_state_filt = filt_out["state_filt"]

    # backward pass
    def backward(state_next, smooth_kwargs):
        mean_state_filt = smooth_kwargs['mean_state_filt']
        var_state_filt = smooth_kwargs['var_state_filt']
        mean_state_pred = smooth_kwargs['mean_state_pred']
        var_state_pred = smooth_kwargs['var_state_pred']
        mean_state_curr, var_state_curr = smooth_mv(
            mean_state_next=state_next["mean"],
            var_state_next=state_next["var"],
            wgt_state=wgt_state,
            mean_state_filt=mean_state_filt,
            var_state_filt=var_state_filt,
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred,
        )
        state_curr = {
            "mean": mean_state_curr,
            "var": var_state_curr
        }
        return state_curr, state_curr
    # initialize
    scan_init = {
        "mean": mean_state_filt[n_steps],
        "var": var_state_filt[n_steps]
    }
    # scan arguments
    scan_kwargs = {
        'mean_state_filt': mean_state_filt[:n_steps-1],
        'var_state_filt': var_state_filt[:n_steps-1],
        'mean_state_pred': mean_state_pred[1:n_steps],
        'var_state_pred': var_state_pred[1:n_steps]
    }
    _, smooth_out = jax.lax.scan(backward, scan_init, scan_kwargs,
                               reverse=True)
    
    # append initial values to back
    mean_state_smooth = jnp.concatenate(
        [smooth_out["mean"], scan_init["mean"][None]]
    )
    var_state_smooth = jnp.concatenate(
        [smooth_out["var"], scan_init["var"][None]]
    )
    return mean_state_smooth, var_state_smooth

def kalman_filter(mean_state_init, var_state_init, 
                  mean_state, wgt_state, var_state,
                  x_meas, mean_meas, wgt_meas, var_meas):
    r"""
    Function to compute the filter and predict mean using a linear state-space model with the Kalman filter.
    """

    n_steps = len(x_meas)
    # forward pass
    def forward(carry, t):
        mean_state_filt, var_state_filt = carry["state_filt"]
        mean_state_pred, var_state_pred, mean_state_next, var_state_next = filter(
                mean_state_past=mean_state_filt,
                var_state_past=var_state_filt,
                mean_state=mean_state,
                wgt_state=wgt_state,
                var_state=var_state,
                x_meas=x_meas[t],
                mean_meas=mean_meas,
                wgt_meas=wgt_meas,
                var_meas=var_meas
        )
        # output
        carry = {
            "state_filt": (mean_state_next, var_state_next)
        }
        stack = {
            "state_filt": (mean_state_next, var_state_next),
            "state_pred": (mean_state_pred, var_state_pred)
        }
        return carry, stack
    
    # scan initial value
    forward_init = {
        "state_filt": (mean_state_init, var_state_init)
    }
    # scan itself
    _, filt_out = jax.lax.scan(forward, forward_init, jnp.arange(n_steps))

    return filt_out


def smooth_sim(x_state_next,
               mean_state_filt,
               var_state_filt,
               mean_state_pred,
               var_state_pred,
               wgt_state):
    r"""
    Perform one step of the Kalman sampling smoother.

    Calculates :math:`\tilde theta_{n|N}` from :math:`\theta_{n|n}`, and :math:`\theta_{n+1|n}`, i.e., :math:`x_{n | N} | x_{n+1 | N} \sim N(\tilde \mu_{n | N}, \tilde \Sigma_{n | N})`.

    Args:
        x_state_next(ndarray(n_sim, n_state)): Simulated state at time n+1 given observations from times[0...N]; denoted by :math:`x_{n+1 | N}`.
        mean_state_filt(ndarray(n_state)): Mean estimate for state at time n given observations from times[0...n]; denoted by :math:`\mu_{n | n}`.
        var_state_filt(ndarray(n_state, n_state)): Covariance of estimate for state at time n given observations from times[0...n]; denoted by :math:`\Sigma_{n | n}`.
        mean_state_pred(ndarray(n_state)): Mean estimate for state at time n+1 given observations from times[0...n]; denoted by :math:`\mu_{n+1 | n}`.
        var_state_pred(ndarray(n_state, n_state)): Covariance of estimate for state at time n+1 given observations from times[0...n]; denoted by :math:`\Sigma_{n+1 | n}`.
        wgt_state(ndarray(n_state, n_state)): Transition matrix defining the solution prior; denoted by :math:`Q_{n+1}`.

    Returns:
        (tuple):
        - **mean_state_sim** (ndarray(n_state)): Mean estimate for state at time n given observations from times[0...N] and :math:`x_{n+1 | N}`; denoted by :math:`\tilde \mu_{n | N}`.
        - **var_state_sim** (ndarray(n_state, n_state)): Covariance of estimate for state at time n given observations from times[0...N] and :math:`x_{n+1 | N}`; denoted by :math;`\tilde \Sigma_{n | N}`.

    """
    var_state_temp, var_state_temp_tilde = _smooth(
        var_state_filt, var_state_pred, wgt_state
    )
    def vmap_fun(x_state):
        mean_state_sim = mean_state_filt + \
            var_state_temp_tilde.dot(x_state - mean_state_pred)
        return mean_state_sim
    mean_state_sim = jax.vmap(vmap_fun)(x_state_next)
    var_state_sim = var_state_filt - \
        var_state_temp_tilde.dot(var_state_temp.T)
    return mean_state_sim, var_state_sim


def flatten_tree(v):
  def f(v):
    leaves, _ = jax.tree_util.tree_flatten(v)
    return jnp.concatenate([x.ravel() for x in leaves])
  out, pullback = jax.vjp(f, v)
  return out, lambda x: pullback(x)[0]

def laplace(key, params, logdensity, latent, observed, n_samples):
    flat_pars, unflatten = flatten_tree(params)
    pars_fisher = jax.hessian(lambda t: logdensity(unflatten(t), latent, observed))(flat_pars)
    pars_var = -jax.scipy.linalg.inv(pars_fisher)
    pars_sample = jax.random.multivariate_normal(
        key=key,
        mean=flat_pars,
        cov=pars_var,
        shape=(n_samples,)
    )
    return pars_sample
