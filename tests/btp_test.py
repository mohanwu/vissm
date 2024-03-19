import jax
import jax.numpy
import scipy as sp
from vissm.block_tridiag import *
jax.config.update("jax_enable_x64", True)

# Build a block tridiagonal matrix
key = jax.random.PRNGKey(0)
key, *subkeys = jax.random.split(key, num=10)

n = 2
A = jax.random.normal(subkeys[0], shape=(n,n))
A = jnp.tril(A)
B = jax.random.normal(subkeys[1], shape=(n,n))
C = jax.random.normal(subkeys[2], shape=(n,n))
C = jnp.tril(C)
D = jax.random.normal(subkeys[3], shape=(n,n))
E = jax.random.normal(subkeys[4], shape=(n,n))
E = jnp.tril(E)
Z = jnp.zeros((n,n))

lowermat = jnp.block([[A, Z, Z],
                      [B, C, Z],
                      [Z, D, E]])
varmat = lowermat.dot(lowermat.T)

varA = varmat[:n,:n]
varB = varmat[n:2*n,:n]
varC = varmat[n:2*n, n:2*n]
varD = varmat[2*n:, n:2*n]
varE = varmat[2*n:, 2*n:]
lower_blocks = jnp.array([varB, varD])
diag_blocks = jnp.array([varA, varC, varE])
lower_chol, diag_chol = btp_chol(lower_blocks, diag_blocks)


lowermat2 = jnp.block([[diag_chol[0], Z, Z],
                        [lower_chol[0], diag_chol[1], Z],
                        [Z, lower_chol[1], diag_chol[2]]])
print(jnp.allclose(lowermat2, jsp.linalg.cholesky(varmat, lower=True)))


uppermat = jnp.block([[diag_chol[0].T, lower_chol[0].T, Z],
                        [Z, diag_chol[1].T, lower_chol[1].T],
                        [Z, Z, diag_chol[2].T]])
print(jnp.allclose(uppermat, lowermat2.T))

# lower_chol = jnp.array([B, D])
# diag_chol = jnp.array([A,C,E])
RVZ = btp_simulate(key, lower_chol, diag_chol)
RVz = jax.random.normal(key, shape=diag_chol.shape[:2])
print(jnp.allclose(jsp.linalg.solve(lowermat2.T, RVz.flatten()), RVZ.flatten()))
print(jnp.allclose(sp.stats.multivariate_normal.entropy(cov=jnp.linalg.inv(varmat)), btp_entropy(diag_chol)))
