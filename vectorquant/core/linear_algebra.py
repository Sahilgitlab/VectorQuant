import math

def zeros(rows, cols=None):
    if cols is None:
        cols = rows
    return [[0.0 for _ in range(cols)] for _ in range(rows)]

def identity(n):
    I = zeros(n, n)
    for i in range(n):
        I[i][i] = 1.0
    return I

def vector_norm(v):
    return math.sqrt(sum(x*x for x in v))

def dot(a, b):
    return sum(x*y for x, y in zip(a, b))

def matrix_add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def matrix_subtract(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def matrix_scale(A, scalar):
    return [[A[i][j] * scalar for j in range(len(A[0]))] for i in range(len(A))]

def matrix_multiply(A, B):
    # Vector-matrix multiplication
    if isinstance(A[0], (int, float)) and isinstance(B[0], list):
        return [sum(A[k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))]
    # Matrix-vector
    if isinstance(A[0], list) and isinstance(B[0], (int, float)):
        return [sum(A[i][k] * B[k] for k in range(len(B))) for i in range(len(A))]
    # Matrix-matrix
    return [[sum(A[i][k] * B[k][j] for k in range(len(B)))
            for j in range(len(B[0]))]
            for i in range(len(A))]

def transpose(A):
    if not A: return []
    if isinstance(A[0], (int, float)):
        return [[x] for x in A]
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]

def lu_decomposition(A):
    n = len(A)
    L = zeros(n, n)
    U = zeros(n, n)
    
    for i in range(n):
        # Upper Triangular
        for k in range(i, n):
            s = sum(L[i][j] * U[j][k] for j in range(i))
            U[i][k] = A[i][k] - s
            
        # Lower Triangular
        for k in range(i, n):
            if i == k:
                L[i][i] = 1.0
            else:
                s = sum(L[k][j] * U[j][i] for j in range(i))
                U_ii = U[i][i]
                if abs(U_ii) < 1e-10:
                    U_ii = 1e-10  # Prevent exact division by zero
                L[k][i] = (A[k][i] - s) / U_ii
                
    return L, U

def solve_lu(L, U, b):
    n = len(L)
    # Forward substitution Ly = b
    y = [0.0] * n
    for i in range(n):
        s = sum(L[i][j] * y[j] for j in range(i))
        y[i] = b[i] - s
        
    # Backward substitution Ux = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(U[i][j] * x[j] for j in range(i + 1, n))
        U_ii = U[i][i] if abs(U[i][i]) > 1e-10 else 1e-10
        x[i] = (y[i] - s) / U_ii
        
    return x

def matrix_inverse(A):
    n = len(A)
    L, U = lu_decomposition(A)
    inv = zeros(n, n)
    for i in range(n):
        # i-th column of identity matrix
        e = [1.0 if j == i else 0.0 for j in range(n)]
        x = solve_lu(L, U, e)
        for j in range(n):
            inv[j][i] = x[j]
    return inv

def determinant(A):
    if len(A) != len(A[0]):
        raise ValueError("Matrix must be square")
    _, U = lu_decomposition(A)
    det = 1.0
    for i in range(len(U)):
        det *= U[i][i]
    return det

def cholesky_decomposition(A):
    n = len(A)
    L = zeros(n, n)
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - s
                L[i][j] = math.sqrt(max(val, 0.0))  # Max to handle small numerical negative zeros
            else:
                L_jj = L[j][j] if L[j][j] > 1e-10 else 1e-10
                L[i][j] = (A[i][j] - s) / L_jj
    return L

def trace(A):
    return sum(A[i][i] for i in range(min(len(A), len(A[0]))))

def qr_decomposition(A):
    # Gram-Schmidt Process
    n = len(A)
    m = len(A[0])
    Q = zeros(n, m)
    R = zeros(m, m)
    
    A_t = transpose(A)
    u_vecs = []
    
    for j in range(m):
        v = A_t[j]
        u = list(v)
        for i in range(j):
            R[i][j] = dot(Q_t[i], v)
            u = [u[k] - R[i][j] * Q_t[i][k] for k in range(n)]
            
        norm_u = vector_norm(u)
        R[j][j] = norm_u
        if norm_u > 1e-10:
            q = [u[k] / norm_u for k in range(n)]
        else:
            q = [0.0] * n
            
        for i in range(n):
            Q[i][j] = q[i]
            
        u_vecs.append(q)
        Q_t = u_vecs
        
    return Q, R

def eigen_decomposition(A, num_simulations=100):
    # QR algorithm for eigenvalues
    # Note: Handles real symmetric matrices well. Complex roots not fully handled.
    n = len(A)
    Ak = [row[:] for row in A]
    Q_total = identity(n)
    
    for _ in range(num_simulations):
        Q, R = qr_decomposition(Ak)
        Ak = matrix_multiply(R, Q)
        Q_total = matrix_multiply(Q_total, Q)
        
    eigenvalues = [Ak[i][i] for i in range(n)]
    eigenvectors = transpose(Q_total)
    
    return eigenvalues, eigenvectors

def pseudoinverse(A):
    # Moore-Penrose pseudoinverse A+ = (A^T A)^-1 A^T
    A_t = transpose(A)
    At_A = matrix_multiply(A_t, A)
    try:
        inv_At_A = matrix_inverse(At_A)
        return matrix_multiply(inv_At_A, A_t)
    except:
        # Fallback for perfectly singular matrices using SVD or simple ridge regression approach
        # A_t A + lambda I
        n = len(At_A)
        ridge = matrix_add(At_A, matrix_scale(identity(n), 1e-6))
        inv_ridge = matrix_inverse(ridge)
        return matrix_multiply(inv_ridge, A_t)

def svd(A):
    # Singular value decomposition
    # Note: Simplified version A = U Sigma V^T
    # via eigendecomposition of A^T A and A A^T
    A_t = transpose(A)
    At_A = matrix_multiply(A_t, A)
    eigenvalues_v, V = eigen_decomposition(At_A)
    
    # Sort eigenvalues and vectors descending
    idx = list(range(len(eigenvalues_v)))
    idx.sort(key=lambda x: eigenvalues_v[x], reverse=True)
    
    singular_values = [math.sqrt(max(eigenvalues_v[i], 0.0)) for i in idx]
    V_sorted = [V[i] for i in idx] # Rows are eigenvectors
    
    U = []
    # U = A V / sigma
    V_t_sorted = transpose(V_sorted) # Cols are eigenvectors
    for i in range(len(singular_values)):
        sigma = singular_values[i]
        v_col = V_sorted[i] # row in V corresponds to col in V^T
        
        Av = matrix_multiply(A, [[x] for x in v_col])
        u = [Av[j][0] / sigma if sigma > 1e-10 else 0.0 for j in range(len(Av))]
        U.append(u)
        
    return transpose(U), singular_values, V_sorted # Return U, Sigma_diag, V^T
