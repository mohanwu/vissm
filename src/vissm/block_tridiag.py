import jax
import jax.numpy as jnp
import jax.scipy as jsp

def btp_chol(lower_blocks, diag_blocks):
    r"""
    Compute the Cholesky decomposition of a symmetric, positive definite block-tridiagonal matrix.

    Args:
        lower_blocks (ndarray(n_blocks-1, n, n)): Lower off-diagonal blocks.
        diag_blocks (ndarray(n_blocks, n, n)): Diagonal blocks.

    Returns:
        (tuple):
        - **lower_chol** (ndarray(n_blocks-1, n, n)): Off-diagonal blocks of the Cholesky.
        - **diag_chol** (ndarray(n_blocks, n, n)): Diagonal blocks of the Cholesky.
    """

    diag0 = jsp.linalg.cholesky(diag_blocks[0], lower=True)
    
    def scan_fun(diag_out, blocks):
        lower_block = blocks["lower"]
        diag_block = blocks["diag"]
        lower = jsp.linalg.solve_triangular(diag_out.T, lower_block.T, trans=1).T
        temp_mat = diag_block - lower.dot(lower.T)
        diag = jsp.linalg.cholesky(temp_mat, lower=True)
        stack ={
            "lower": lower,
            "diag": diag
        }
        return diag, stack
    
    blocks = {
        "lower": lower_blocks,
        "diag": diag_blocks[1:]
    }

    _, scan_out = jax.lax.scan(scan_fun, diag0, blocks)
    diag_chol = jnp.concatenate([diag0[None], scan_out["diag"]])
    lower_chol = scan_out["lower"]
    return lower_chol, diag_chol


def btp_simulate(key, lower_blocks, diag_blocks, n_sim=1):
    r"""
    Simulate multivariate normal assuming that a block-bidiagonal Cholesky factor of the inverse covariance matrix.
    Let :math:`L` be the Cholesky factor constructed by the blocks, then simulate :math:`Z \sim N(0, L^{-T} L^{-1})`
    
    Args:
        lower_blocks (ndarray(n_blocks-1, n, n)): Lower off-diagonal blocks of the Cholesky factor of the inverse covariance.
        diag_blocks (ndarray(n_blocks, n, n)): Diagonal blocks of the Cholesky factor of the inverse covariance.

    Returns:
        (ndarray(n_blocks, n)): Simulated multivariate normals.
    """
    n_blocks, n = diag_blocks.shape[:2]
    z = jax.random.normal(key, shape=(n_blocks, n, n_sim))
    upper_blocks = lower_blocks.transpose(0,2,1)
    udiag_blocks = diag_blocks.transpose(0,2,1)
    ZN = jsp.linalg.solve_triangular(udiag_blocks[-1], z[-1])
    def scan_fun(Z, t):
        temp_mat = z[t] - upper_blocks[t].dot(Z)
        Z = jsp.linalg.solve_triangular(udiag_blocks[t], temp_mat)
        return Z, Z
    
    _, scan_out = jax.lax.scan(scan_fun, ZN, jnp.arange(len(upper_blocks)),
                               reverse=True)
    Z = jnp.concatenate([scan_out, ZN[None]])
    return Z


def btp_entropy(diag_blocks):
    r""""
    Calculate the entropy :math:`E(-\log q(x|y))` for a multivariate normal given the Cholesky of the inverse covariance matrix.

    Args:
        diag_blocks (ndarray(n_blocks, n, n)): Diagonal blocks of the Cholesky factor of the inverse covariance.
    
    Returns:
        (float): The entropy of the multivariate normal.
    """
    n_blocks, n = diag_blocks.shape[:2]
    def scan_fun(half_det, diag_block):
        half_det -= jnp.sum(jnp.log(jnp.diag(diag_block)))
        return half_det, None

    half_det, _ = jax.lax.scan(scan_fun, 0.0, diag_blocks)
    return half_det + n_blocks*n/2.0*(1+jnp.log(2*jnp.pi))


def btp_var(lower_blocks, diag_blocks):
    r"""
    Let :math:`L` be the Cholesky factor constructed by the blocks, then compute the diagonal of :math:`L^{-T} L^{-1})`
    
    Args:
        lower_blocks (ndarray(n_blocks-1, n, n)): Lower off-diagonal blocks of the Cholesky factor of the inverse covariance.
        diag_blocks (ndarray(n_blocks, n, n)): Diagonal blocks of the Cholesky factor of the inverse covariance.

    Returns:
        (ndarray(n_blocks, n)): Variance.
    """
    V1 = diag_blocks[0].dot(diag_blocks[0].T)
    Vn = jax.vmap(lambda x, y: x.dot(x.T) + y.dot(y.T))(
        lower_blocks,
        diag_blocks[1:]
    )
    V_out = jnp.concatenate([V1[None], Vn])
    return V_out

def btp_logpdf(x, lower_blocks, diag_blocks):
    r"""
    Compute the logpdf of a multivariate distribution with 0 mean given the inverse bi-diagonal Cholesky.
    
    Args:
        x (ndarray(n_blocks, n)): Observations in blocks.
        lower_blocks (ndarray(n_blocks-1, n, n)): Lower off-diagonal blocks of the Cholesky factor of the inverse covariance.
        diag_blocks (ndarray(n_blocks, n, n)): Diagonal blocks of the Cholesky factor of the inverse covariance.

    Returns:
        (float): Logpdf.

    """
    n_blocks, n = x.shape
    x = jnp.vstack((x, jnp.zeros((1, n))))
    
    def scan_fun(logpdf, args):
        vec = args["vec"]
        vec_next = args["vec_next"]
        lower_block = args["lower_block"]
        diag_block = args["diag_block"]
        half = vec.T.dot(diag_block) + vec_next.T.dot(lower_block)
        logpdf -= 0.5*(half.dot(half.T))
        logpdf += jnp.sum(jnp.log(jnp.diag(diag_block)))
        return logpdf, logpdf
    
    scan_init = {
        "vec": x[:-1],
        "vec_next": x[1:],
        "lower_block": jnp.vstack((lower_blocks, jnp.zeros((1,n,n)))),
        "diag_block": diag_blocks
    }
    log_pdf, _ = jax.lax.scan(scan_fun, 0.0, scan_init)
    const = -n_blocks*n*0.5*jnp.log(2*jnp.pi)
    return log_pdf + const
