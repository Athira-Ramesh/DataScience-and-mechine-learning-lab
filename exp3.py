import numpy as np
mat=np.array([[5, 6, 4],[4, 5, 6],[3, 5, 6]])
U, S, VT = np.linalg.svd(mat)
print("U matrix:\n",U)
print("S matrix:\n",np.diag(S))
print("VT\n",VT)
reconstructed_mat=np.dot(U,np.dot(np.diag(S),VT))
print("reconstructed_mat",reconstructed_mat)