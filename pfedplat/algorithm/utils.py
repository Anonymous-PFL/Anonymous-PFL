import torch
import numpy as np
import cvxopt
from cvxopt import matrix
from scipy.optimize import minimize
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):

    P = 0.5 * (P + P.T)  

    P = P.astype(np.double)
    q = q.astype(np.double)
    args = [matrix(P), matrix(q)]  
    if G is not None:
        args.extend([matrix(G), matrix(h)])  
        if A is not None:
            args.extend([matrix(A), matrix(b)])  
    sol = cvxopt.solvers.qp(*args)
    optimal_flag = 1
    if 'optimal' not in sol['status']:
        optimal_flag = 0
    return np.array(sol['x']).reshape((P.shape[1],)), optimal_flag


def setup_qp_and_solve(vec):
    
    P = np.dot(vec, vec.T)

    n = P.shape[0]
    q = np.zeros(n)

    G = - np.eye(n)
    h = np.zeros(n)

    A = np.ones((1, n))
    b = np.ones(1)

    cvxopt.solvers.options['show_progress'] = False

    sol, optimal_flag = cvxopt_solve_qp(P, q, G, h, A, b)
    return sol, optimal_flag


def setup_qp_and_solve_for_mgdaplus(vec, epsilon, lambda0):
    
    P = np.dot(vec, vec.T)

    n = P.shape[0]
    q = np.zeros(n)

    
    G = np.vstack([-np.eye(n), np.eye(n)])
    lb = np.array([max(0, lambda0[i] - epsilon) for i in range(n)])
    ub = np.array([min(1, lambda0[i] + epsilon) for i in range(n)])
    h = np.hstack([lb, ub])

    A = np.ones((1, n))
    b = np.ones(1)

    cvxopt.solvers.options['show_progress'] = False
    sol, optimal_flag = cvxopt_solve_qp(P, q, G, h, A, b)
    return sol, optimal_flag


def quadprog(P, q, G, h, A, b):
    P = cvxopt.matrix(P.tolist())
    q = cvxopt.matrix(q.tolist(), tc='d')
    G = cvxopt.matrix(G.tolist())
    h = cvxopt.matrix(h.tolist())
    A = cvxopt.matrix(A.tolist())
    b = cvxopt.matrix(b.tolist(), tc='d')
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(P, q.T, G.T, h.T, A.T, b)
    return np.array(sol['x'])


def setup_qp_and_solve_for_mgdaplus_1(vec, epsilon, lambda0):
    
    P = np.dot(vec, vec.T)

    n = P.shape[0]

    q = np.array([[0] for i in range(n)])
    
    A = np.ones(n).T
    b = np.array([1])
    
    lb = np.array([max(0, lambda0[i] - epsilon) for i in range(n)])
    ub = np.array([min(1, lambda0[i] + epsilon) for i in range(n)])
    G = np.zeros((2 * n, n))
    for i in range(n):
        G[i][i] = -1
        G[n + i][i] = 1
    h = np.zeros((2 * n, 1))
    for i in range(n):
        h[i] = -lb[i]
        h[n + i] = ub[i]
    sol = quadprog(P, q, G, h, A, b).reshape(-1)

    return sol, 1


def get_d_moomtl_d(grads, device):
    """ calculate the gradient direction for FedMGDA """

    vec = grads
    sol, optimal_flag = setup_qp_and_solve(vec.cpu().detach().numpy())  

    sol = torch.from_numpy(sol).float().to(device)
    
    d = torch.matmul(sol, grads)

    
    descent_flag = 1
    c = - (grads @ d)

    if not torch.all(c <= 1e-6):
        descent_flag = 0

    return d, optimal_flag, descent_flag


def get_d_mgdaplus_d(grads, device, epsilon, lambda0):
    """ calculate the gradient direction for FedMGDA+ """

    vec = grads
    sol, optimal_flag = setup_qp_and_solve_for_mgdaplus_1(vec.cpu().detach().numpy(), epsilon, lambda0)
    

    sol = torch.from_numpy(sol).float().to(device)
    d = torch.matmul(sol, grads)

    
    descent_flag = 1
    c = -(grads @ d)
    if not torch.all(c <= 1e-6):
        descent_flag = 0
        
        
    
    return d, optimal_flag, descent_flag


def check_constraints(value, ref_vec, prefer_vec):
    
    w = ref_vec - prefer_vec

    gx = torch.matmul(w, value/torch.norm(value))
    idx = gx > 0
    return torch.sum(idx), idx


def project(a, b):
    return a @ b / torch.norm(b)**2 * b


def solve_d(Q, g, value, device):
    L = value.cpu().detach().numpy()
    QTg = Q @ g.T
    QTg = QTg.cpu().detach().numpy()
    gTg = g @ g.T
    gTg = gTg.cpu().detach().numpy()
    def fun(x):
        return np.sum((gTg @ x - L)**2)

    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  
            {'type': 'ineq', 'fun': lambda x: x - 1e-10},  
            {'type': 'ineq', 'fun': lambda x: QTg @ x}  
            )

    x0 = np.random.rand(g.shape[0])
    x0 = x0 / np.sum(x0)
    res = minimize(fun, x0, method='SLSQP', constraints=cons)
    lam = res.x
    lam = torch.from_numpy(lam).float().to(device)
    d = lam @ g
    return d


def get_fedmgdp_d(grads, value, add_grads, alpha, prefer_vec, force_active, device):
    """ calculate the gradient direction for FedMGDP """
    """传入格式：grads: 矩阵，行数等于个体数，列数等于网络参数总数"""

    value_norm = torch.norm(value)
    norm_values = value / value_norm
    prefer_vec /= torch.norm(prefer_vec)

    
    cos = float(norm_values @ prefer_vec)
    cos = min(1, cos)  
    cos = max(-1, cos)  
    bias = np.arccos(cos) / np.pi * 180
    pref_active_flag = (bias > alpha) | force_active

    if not pref_active_flag:
        vec = grads
        pref_active_flag = 0

    else:
        pref_active_flag = 1

        
        
        
        

        
        h_vec = (prefer_vec @ norm_values * norm_values - prefer_vec).reshape(1, -1)
        h_vec /= torch.norm(h_vec)  
        fair_grad = h_vec @ grads

        vec = torch.cat((grads, fair_grad))

    if add_grads is not None:
        vec = torch.vstack([vec, add_grads])

    sol, optimal_flag = setup_qp_and_solve(vec.cpu().detach().numpy())  
    sol = torch.from_numpy(sol).float().to(device)
    
    
    d = sol @ vec  

    
    descent_flag = 1  
    c = - (vec @ d)
    print('c:', torch.max(c))
    print(torch.min(c))
    
    if not torch.all(c <= 1e-6):
        descent_flag = 0
        
        
        

    return d, vec, pref_active_flag, optimal_flag, descent_flag


def get_fedmdfg_d_layers(grads, value, add_grads, alpha, prefer_vec, force_active, Loc_list, device):
    """
    分层MGDA
    传入格式：grads: 矩阵，行数等于个体数，列数等于网络参数总数。
    """

    value_norm = torch.norm(value)
    norm_values = value / value_norm
    prefer_vec /= torch.norm(prefer_vec)

    
    cos = float(norm_values @ prefer_vec)
    cos = min(1, cos)  
    cos = max(-1, cos)  
    bias = np.arccos(cos) / np.pi * 180
    pref_active_flag = (bias > alpha) | force_active

    if not pref_active_flag:
        vec = grads
        pref_active_flag = 0

    else:
        pref_active_flag = 1

        
        
        
        

        
        h_vec = (prefer_vec @ norm_values * norm_values - prefer_vec).reshape(1, -1)
        h_vec /= torch.norm(h_vec)  
        fair_grad = h_vec @ grads

        vec = torch.cat((grads, fair_grad))

    if add_grads is not None:
        vec = torch.vstack([vec, add_grads])

    
    
    
    
    
    d = []
    for i in range(len(Loc_list)):
        vec_layer = vec[:, Loc_list[i]]
        sol_layer, _ = setup_qp_and_solve(vec_layer.cpu().detach().numpy())
        sol_layer = torch.from_numpy(sol_layer).float().to(device)
        d_layer = sol_layer @ vec_layer
        d.append(d_layer)
    d = torch.hstack(d)

    
    
    base_indices = np.hstack([Loc_list[i] for i in range(len(Loc_list) - 1)]).tolist()
    vec_base = vec[:, base_indices]
    sol_base, _ = setup_qp_and_solve(vec_base.cpu().detach().numpy())
    sol_base = torch.from_numpy(sol_base).float().to(device)
    d_base = sol_base @ vec_base
    
    vec_head = vec[:, Loc_list[-1]]
    sol_head, _ = setup_qp_and_solve(vec_head.cpu().detach().numpy())
    sol_head = torch.from_numpy(sol_head).float().to(device)
    d_head = sol_head @ vec_head
    d = torch.hstack([d_base, d_head])

    
    descent_flag = 1  
    c = - (vec @ d)
    print('c:', torch.max(c))
    print(torch.min(c))
    
    if not torch.all(c <= 1e-6):
        descent_flag = 0
        
        
        

    return d, vec, pref_active_flag, 1, descent_flag


def get_GPFL_d(grads, value, add_grads, prefer_vec, miu, device):
    """ calculate the gradient direction for GPFL
    传入格式：grads: 矩阵，行数等于个体数，列数等于网络参数总数。
    """

    value_norm = torch.norm(value)
    value = value / value_norm

    Q = grads
    
    if add_grads is not None:
        add_grads = add_grads / torch.norm(add_grads, dim=1).reshape(-1, 1) * miu
        Q = torch.vstack([Q, add_grads])
    g = Q  
    
    h_vec = (prefer_vec @ value * value - prefer_vec).reshape(1, -1)
    fair_grad = h_vec @ grads
    fair_grad = fair_grad / torch.norm(fair_grad) * miu
    Q = torch.cat((Q, fair_grad))

    
    sol, optimal_flag = setup_qp_and_solve(Q.cpu().detach().numpy())  
    sol = torch.from_numpy(sol).float().to(device)
    d = sol @ Q

    
    descent_flag = 1  
    c = - (Q @ d)
    print('c:', torch.max(c))
    print(torch.min(c))
    
    if not torch.all(c <= 1e-6):
        descent_flag = 0
        
        
        

    return d, Q, g, fair_grad

def get_GPFL_d_new_p(grads, value, add_grads, prefer_vec, miu, device):
    """ calculate the gradient direction for GPFL
    传入格式：grads: 矩阵，行数等于个体数，列数等于网络参数总数。
    """

    value_norm = torch.norm(value)
    value = value / value_norm

    Q = grads
    
    if add_grads is not None:
        add_grads = add_grads / torch.norm(add_grads, dim=1).reshape(-1, 1) * miu
        Q = torch.vstack([Q, add_grads])
    g = Q  
    
    h_vec = (prefer_vec @ value * value - prefer_vec).reshape(1, -1)
    fair_grad = h_vec @ grads
    fair_grad = fair_grad / torch.norm(fair_grad) * miu
    Q = torch.cat((Q, fair_grad))

    
    sol, optimal_flag = setup_qp_and_solve(Q.cpu().detach().numpy())  
    sol = torch.from_numpy(sol).float().to(device)
    d = sol @ Q

    
    descent_flag = 1  
    c = - (Q @ d)
    print('c:', torch.max(c))
    print(torch.min(c))
    
    if not torch.all(c <= 1e-6):
        descent_flag = 0
        
        
        

    return d, Q, g, h_vec


def get_GPFL_d_mgda(grads, value, add_grads, prefer_vec, device):

    value_norm = torch.norm(value)
    value = value / value_norm
    prefer_vec /= torch.norm(prefer_vec)

    Q = grads
    
    if add_grads is not None:
        Q = torch.vstack([Q, add_grads])
    g = Q  
    
    h_vec = (prefer_vec @ value * value - prefer_vec).reshape(1, -1)
    h_vec /= torch.norm(h_vec)  
    fair_grad = h_vec @ grads
    Q = torch.cat((Q, fair_grad))

    
    
    sol, optimal_flag = setup_qp_and_solve(Q.cpu().detach().numpy())  
    sol = torch.from_numpy(sol).float().to(device)
    g_f = sol @ Q
    
    sol, optimal_flag = setup_qp_and_solve(g.cpu().detach().numpy())  
    sol = torch.from_numpy(sol).float().to(device)
    g_d = sol @ g

    
    cos = float(value @ prefer_vec)
    cos = min(1, cos)  
    cos = max(-1, cos)  
    angle = np.arccos(cos)
    r = (np.pi/2 - 0.5 * (np.pi/2 - angle) * 4 * (np.pi/2 - angle)/np.pi) / (np.pi/2)
    d = (g_f - g_d) * r + g_d

    
    descent_flag = 1  
    c = - (Q @ d)
    print('c:', torch.max(c))
    print(torch.min(c))
    
    if not torch.all(c <= 1e-6):
        descent_flag = 0
        
        
        

    return d, Q, g


def get_GPFL_d_layers(grads, value, add_grads, prefer_vec, Loc_list, device):
    """ calculate the gradient direction for GPFL
    传入格式：grads: 矩阵，行数等于个体数，列数等于网络参数总数。
    """

    value_norm = torch.norm(value)
    norm_values = value / value_norm
    prefer_vec /= torch.norm(prefer_vec)

    Q = grads
    
    if add_grads is not None:
        Q = torch.vstack([Q, add_grads])
    g = Q  
    
    h_vec = (prefer_vec @ norm_values * norm_values - prefer_vec).reshape(1, -1)
    h_vec /= torch.norm(h_vec)  
    fair_grad = h_vec @ grads
    Q = torch.cat((Q, fair_grad))

    
    
    
    
    
    
    
    
    

    
    
    base_indices = np.hstack([Loc_list[i] for i in range(len(Loc_list) - 1)]).tolist()
    Q_base = Q[:, base_indices]
    sol_base, _ = setup_qp_and_solve(Q_base.cpu().detach().numpy())
    sol_base = torch.from_numpy(sol_base).float().to(device)
    d_base = sol_base @ Q_base
    
    Q_head = Q[:, Loc_list[-1]]
    sol_head, _ = setup_qp_and_solve(Q_head.cpu().detach().numpy())
    sol_head = torch.from_numpy(sol_head).float().to(device)
    d_head = sol_head @ Q_head
    d = torch.hstack([d_base, d_head])

    
    descent_flag = 1  
    c = - (Q @ d)
    print('c:', torch.max(c))
    print(torch.min(c))
    
    if not torch.all(c <= 1e-6):
        descent_flag = 0
        
        
        

    return d, Q, g

def get_GPFL_d_layers_each(grads, value, add_grads, prefer_vec, miu, Loc_list, device):
    """ 逐层分层 calculate the gradient direction for GPFL
    传入格式：grads: 矩阵，行数等于个体数，列数等于网络参数总数。
    """

    value_norm = torch.norm(value)
    value = value / value_norm
    prefer_vec = prefer_vec / torch.norm(prefer_vec)

    Q = grads
    
    if add_grads is not None:
        add_grads = add_grads / torch.norm(add_grads, dim=1).reshape(-1, 1) * miu
        Q = torch.vstack([Q, add_grads])
    g = Q  
    
    h_vec = (prefer_vec @ value * value - prefer_vec).reshape(1, -1)
    fair_grad = h_vec @ grads
    fair_grad = fair_grad / torch.norm(fair_grad) * miu
    Q = torch.cat((Q, fair_grad))

    
    d = []
    for i in range(len(Loc_list)):
        Q_layer = Q[:, Loc_list[i]]
        sol_layer, _ = setup_qp_and_solve(Q_layer.cpu().detach().numpy())
        sol_layer = torch.from_numpy(sol_layer).float().to(device)
        d_layer = sol_layer @ Q_layer
        d.append(d_layer)
    d = torch.hstack(d)

    
    descent_flag = 1  
    c = - (Q @ d)
    print('c:', torch.max(c))
    print(torch.min(c))
    
    if not torch.all(c <= 1e-6):
        descent_flag = 0
        
        
        

    return d, Q, g
