import numpy as np

def compute_transmat(P, pi=None):
    if pi is None:
        # set pi to be uniform policy
        pi = np.ones((P.shape[1], P.shape[0])) / P.shape[0]
    return np.diagonal(np.tensordot(pi, P, axes=1), axis1=0, axis2=1).T

def compute_succrep(transmat, discount):
    return np.linalg.inv(np.eye(transmat.shape[0]) - discount * transmat)

def signed_amp(x):
    """Return sign(x) * amp(x), where amp is amplitude of complex number"""
    return np.sign(np.real(x)) * np.sqrt(np.real(x) ** 2 + np.imag(x) ** 2)

def eig(x, order="descend", sortby=signed_amp):
  """Computes eigenvectors and returns them in eigenvalue order.

  Args:
    x: square matrix to eigendecompose
    order: "descend" or "ascend" to specify in which order to sort eigenvalues
      (default="descend")
    sortby: function transforms a list of (possibly complex, possibly mixed
      sign) into real-valued scalars that can be sorted without ambiguity
      (default=signed_amp)

  Returns:
    evecs: array of eigenvectors
    evals: matrix with eigenvector columns
  """
  assert x.shape[0] == x.shape[1]
  n = x.shape[0]
  evals, evecs = np.linalg.eig(x)

  ind_order = range(n)
  ind_order = [x for _, x in sorted(zip(sortby(evals), ind_order))]
  if order == "descend":
    ind_order = ind_order[::-1]
  evals = evals[ind_order]
  evecs = evecs[:, ind_order]
  return evecs, evals
