import numpy as np
import sys

class ELBMF:
    def __init__ (  self,
                    org_A,
                    A,
                    ncomponents,
                    l1reg,
                    l2reg,
                    c, # = t -> c^t 
                    maxiter,
                    tolerance,
                    random_seed          = 19,
                    beta                 = 0.0, # inertial disabled by default
                    batchsize            = None,
                    with_rounding        = True,
                    callback             = None):
        self.org_A = org_A.copy()
        self.A = A.copy()
        self.n, self.m = np.shape(A)
        self.k = ncomponents
        self.l1reg = l1reg
        self.l2reg = l2reg
        self.c = c
        self.maxiter = maxiter
        self.tolerance = tolerance
        self.random_seed = random_seed
        self.beta = beta
        if(batchsize is  None):
            self.batchsize  = len(A)
        else:
            self.batchsize = batchsize
        self.with_rounding = with_rounding
        self.callback = callback
        self.U, self.V = self.init_factorization(random_seed)
        U_init = _rounding(self.U)
        V_init = _rounding(self.V)
        print('init_loss')
        _print_loss(org_A, U_init, V_init)
    
    def init_factorization(self, seed):
        np.random.seed(seed)
        return np.random.rand(self.n,self.k), np.random.rand(self.k,self.m)
    
    def factorize(self):
        if(self.batchsize >= len(self.A)):
            if(self.beta == 0):
                self.U, self.V = _factorize_palm(self)
            else:
                self.U, self.V = _factorize_ipalm(self, self.beta)
        else:
            self.U, self.V = _batched_factorize_ipalm(self)
            
        self.U = _rounding(self.U)
        self.V = _rounding(self.V)
        print('result_loss')
        self.print_loss()
        
    def print_loss(self):
        _print_loss(self.org_A, self.U, self.V)
            
# def _regularization_rate(c, t):
#     return pow(c, t)
def _product(X, Y):
    return np.clip(np.dot(X, Y), 0, 1)

def _print_loss(A, U, V):
    print(f'recall : {_recall(A, U, V)}, similarity : {_similarity(A, U, V)}, relative loss : {_relative_loss(A, U, V)}', flush=True)
        
def _recall(A, U, V):
    return np.sum(A * _product(U, V))/np.sum(A)

def _similarity(A, U, V):
    return np.sum(np.abs(A - _product(U, V)))/A.size

def _relative_loss(A, U, V):
    return np.sum(np.abs(A - _product(U, V)))*np.sum(A)

def _rounding(X, k = 0.5, l = 1.e+20):
    X = _prox(X, k, l)
    return np.round(np.clip(X, 0, 1))

def _proxel_1(X, k, l):
    tmp = np.zeros_like(X)
    tmp[X <=0.5] = (X[X <=0.5] - k * np.sign(X[X <=0.5])) / (1 + l)
    tmp[X >0.5] = (X[X > 0.5] - k * np.sign(X[X > 0.5] - 1) + l) / (1 + l)
    return tmp

def _prox(U, k, l):
    prox_U = _proxel_1(U, k, l)
    prox_U[prox_U<0] = 0
    return prox_U

def  _reducemf_impl(A, U, V, l1reg, l2reg):
    VVt =  np.dot(V,V.T)
    grad_u = np.dot(U, VVt) - np.dot(A, V.T)
    L = max(np.linalg.norm(VVt, ord=2), 1e-4)
    step_size = 1 / (1.1 * L)
    U = U - grad_u * step_size
    U = _prox(U, l1reg*step_size, l2reg*step_size)
    
    return U

def  _reducemf_impl_b(A, U, V, l1reg, l2reg, U_, beta):
    U__ = U.copy()
    U = U + (U - U_) * beta 
    VVt =  np.dot(V,V.T)
    grad_u = np.dot(U, VVt) - np.dot(A, V.T)
    L = max(np.linalg.norm(VVt, ord=2), 1e-4)
    step_size = 2 * (1 - beta) / (1 + 2 * beta) / L
    U = U - grad_u * step_size
    U = _prox(U, l1reg*step_size, l2reg*step_size)
    
    return U, U__
    
                   
def _factorize_palm(elbmf):
    U = elbmf.U
    V = elbmf.V
    l2reg_init = elbmf.l2reg
    l1reg = elbmf.l1reg
    c = elbmf.c
    A = elbmf.A
    tol = elbmf.tolerance
    ell0 = sys.float_info.max
    ell = 0
    for iter in range(elbmf.maxiter):
        l2reg = l2reg_init * pow(c, iter)
        U = _reducemf_impl(A, U, V, l1reg, l2reg)
        V = _reducemf_impl(A.T, V.T, U.T, l1reg, l2reg).T
        ell = np.linalg.norm(A - _product(U,V), ord=2)
        if(abs(ell - ell0) < tol): break
        ell0 = ell
    return U, V

def _factorize_ipalm(elbmf, beta):
    U = elbmf.U
    V = elbmf.V
    l2reg_init = elbmf.l2reg
    l1reg = elbmf.l1reg
    c = elbmf.c
    A = elbmf.A
    tol = elbmf.tolerance
    ell0 = sys.float_info.max
    ell = 0
    U_    = U.copy()
    Vt_   = V.T.copy()
    for iter in range(elbmf.maxiter):
        l2reg = l2reg_init * pow(c, iter)
        U, U_ = _reducemf_impl_b(A, U, V, l1reg, l2reg, U_, beta)
        Vt, Vt_ = _reducemf_impl_b(A.T, V.T, U.T, l1reg, l2reg, Vt_, beta)
        V = Vt.T.copy()
        ell = np.linalg.norm(A - _product(U,V), ord=2)
        if(abs(ell - ell0) < tol): break
        ell0 = ell
    return U, V

def _batched_factorize_ipalm(elbmf):
    # do this later
    U = elbmf.U.copy()
    V = elbmf.V.copy()
    
    return U, V
