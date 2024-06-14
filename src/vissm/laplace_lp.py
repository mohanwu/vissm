import jax
import jax.numpy as jnp

def laplace_lp(theta, y_meas, x_init, joint_neglogdensity, solver):
    r"""
    We use the Laplace approximation to find `theta_opt`. The procedure is as follows:
    
    1. Use the identity $p(y|\theta) = p(y,x|\theta) p(x|y,\theta)$
    2. Approximate $p(x|y, \theta)$ by the mode-quadrature normal with mean $\hat x = \argmax p(y, x|\theta)$
       and variance equal to the negative inverse Hessian.
    2. Compute 1. using $x= \hat x$. 
    
    Args:
        theta: Vector of model parameters.
        y_meas: Observations.
        x_init: Starting value to find $\hat x$.
        joint_neglogdensity: Equal to $-p(y,x|\theta)$.
        solver: Optimization method for finding the minimum.

    Returns:
        The logdensity of the Laplace approximation. 
    """
    # find \hat x and associated Hessian
    n_seq, n_state = x_init.shape
    x_opt = solver.run(x_init, y_meas, theta).params
    # x_opt = x_init
    x_hes = jax.hessian(joint_neglogdensity)(x_opt, y_meas, theta).reshape(n_seq*n_state, n_seq*n_state)
    x_var = jnp.linalg.inv(x_hes)
    # approximate conditional by MVN
    cond_lp = jax.scipy.stats.multivariate_normal.logpdf(x_opt.flatten(), x_opt.flatten(), x_var)
    # return x_var
    return -joint_neglogdensity(x_opt, y_meas, theta) - cond_lp

