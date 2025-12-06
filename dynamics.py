import sympy as sp
import numpy as np
from sympy.utilities.lambdify import lambdify

#   Symbols 
I1, I2 = sp.symbols('I1 I2')
l1, lc1, l2, lc2 = sp.symbols('l1 lc1 l2 lc2')
m1, m2 = sp.symbols('m1 m2')
theta1, theta2 = sp.symbols('theta1 theta2')
theta1_dot, theta2_dot = sp.symbols('theta1_dot theta2_dot')
g = sp.symbols('g')
f1, f2 = sp.symbols('f1 f2')
tau1, tau2 = sp.symbols('tau1 tau2') # Define input symbols for Jacobian calculation

params_1 = {
    m1: 1.0,
    m2: 1.0,
    l1: 1.0, 
    lc1: 1.0/2,
    l2: 1.0,
    lc2: 1.0/2,
    # r1: 0.5,
    # r2: 0.5,
    I1: 0.33,
    I2: 0.33,
    g: 9.81,
    f1: 1.0,
    f2: 1.0
}

params_2 = {
    m1: 2.0,
    m2: 2.0,
    l1: 1.5, 
    lc1: 1.5/2,
    l2: 1.5,
    lc2: 1.5/2,
    # r1: 0.75,
    # r2: 0.75,
    I1: 1.5,
    I2: 1.5,
    g: 9.81,
    f1: 1.0,
    f2: 1.0
}

params_3 = {
    m1: 1.5,
    m2: 1.5,
    l1: 2.0,
    lc1: 2.0/2, 
    l2: 2.0,
    lc2: 2.0/2,
    # r1: 1.0,
    # r2: 1.0,
    I1: 2.0,
    I2: 2.0,
    g: 9.81,
    f1: 1.0,
    f2: 1.0
}

#   matrix M 
M11 = I1 + I2 + lc1**2 * m1 + m2*(l1**2 + 2*l1*lc2*sp.cos(theta2) + lc2**2)
M12 = I2 + lc2*m2*(l1*sp.cos(theta2) + lc2)
M21 = M12
M22 = I2 + lc2**2 * m2

M = sp.Matrix([[sp.simplify(M11), sp.simplify(M12)],
               [sp.simplify(M21), sp.simplify(M22)]])

#   matrix C 
C11 = - l1*lc2*m2*theta2_dot*sp.sin(theta2)
C12 = - l1*lc2*m2*(theta1_dot + theta2_dot)*sp.sin(theta2)
C21 = l1*lc2*m2*theta1_dot*sp.sin(theta2)
C22 = 0

C = sp.Matrix([[C11, C12],
               [C21, C22]])

#   vector G 
G1 = g*lc1*m1*sp.sin(theta1) + g*m2*(l1*sp.sin(theta1) + lc2*sp.sin(theta1 + theta2))
G2 = g*m2*lc2*sp.sin(theta1 + theta2)

Gvec = sp.Matrix([[sp.simplify(G1)],
                  [sp.simplify(G2)]])

#   matrix F
F = sp.Matrix([[f1, 0],
               [0, f2]])


# Assign set 1 of parameters as default
M_num = M.subs(params_1)
C_num = C.subs(params_1)
G_num = Gvec.subs(params_1)
F_num = F.subs(params_1)

q_syms = [theta1, theta2]
qdot_syms = [theta1_dot, theta2_dot]
all_syms = q_syms + qdot_syms
u_syms = [sp.symbols('tau1'), sp.symbols('tau2')]
x_syms = q_syms + qdot_syms

tau_vec = sp.Matrix(u_syms)
qdot_vec = sp.Matrix(qdot_syms)

# ddq = M^{-1} * (tau - (C+F) qdot - G)
RHS_expr = tau_vec - (C_num @ qdot_vec + F_num @ qdot_vec + G_num)
qddot_expr = M.inv() @ RHS_expr
f_expr = sp.Matrix(qdot_syms + list(qddot_expr))   # [qdot; qddot]
M_func = lambdify(q_syms, M_num, 'numpy')
RHS_func = lambdify(all_syms + u_syms, RHS_expr, 'numpy')
A_expr = f_expr.jacobian(x_syms)
B_expr = f_expr.jacobian(u_syms)

def set_params(version_num):
    # Set the parameters to one of the predefined set
    # Accept values from 1-3
    
    if version_num == 1:
        M_num = M.subs(params_1)
        C_num = C.subs(params_1)
        G_num = Gvec.subs(params_1)
        F_num = F.subs(params_1)
    elif version_num == 2:
        M_num = M.subs(params_2)
        C_num = C.subs(params_2)
        G_num = Gvec.subs(params_2)
        F_num = F.subs(params_2)
    elif version_num == 3:
        M_num = M.subs(params_3)
        C_num = C.subs(params_3)
        G_num = Gvec.subs(params_3)
        F_num = F.subs(params_3)
    else:
        print("Invalid parameter version number, setting the default one.")
        
        M_num = M.subs(params_1)
        C_num = C.subs(params_1)
        G_num = Gvec.subs(params_1)
        F_num = F.subs(params_1)
    
    return M_num, C_num, G_num, F_num
        
##  Task 2 Set up   ##
M_sym, C_sym, G_sym, F_sym = set_params(1) #Assuming 1st config

q_vec = sp.Matrix([theta1, theta2])
qdot_vec = sp.Matrix([theta1_dot, theta2_dot])
x_vec = sp.Matrix([theta1, theta2, theta1_dot, theta2_dot])
u_vec = sp.Matrix([tau1, tau2])  # nu = 2
tau_acrobot = sp.Matrix([0, tau2])

# Dynamics Equation: M * qddot + C * qdot + F * qdot + G = tau
# qddot = M_inv * (tau - C*qdot - F*qdot - G)
RHS_sym = tau_acrobot - ((C_sym + F_sym) @ qdot_vec + G_sym)
qddot_sym = M_sym.LUsolve(RHS_sym) # Symbolic solve

f_cont_sym = sp.Matrix.vstack(qdot_vec, qddot_sym)

# Compute Jacobians ONCE symbolically
A_sym = f_cont_sym.jacobian(x_vec)
B_sym = f_cont_sym.jacobian(u_vec) # Jacobian w.r.t the scalar input

# Generate Numerical Functions (Global)
# These are fast compiled numpy functions
calc_continuous_dynamics = lambdify(list(x_vec) + list(u_vec), f_cont_sym, 'numpy')
func_A = lambdify(list(x_vec) + list(u_vec), A_sym, 'numpy')
func_B = lambdify(list(x_vec) + list(u_vec), B_sym, 'numpy')


dt = 1e-2   # discretization step -> changed from 1e-3 to be faster
ns = 4      # number of states
ni = 2      # number of inputs

def dynamics(xx, uu):
    # Discrete dynamics of gymnast
    # Using Runge-Kutta 4th order
    
    xx = xx.squeeze()
    uu = uu.squeeze()
    
    xx_next = np.zeros((ns,))
    
    # Runge-Kutta 4th order
    k1 = continuous_dynamics(xx, uu)
    k2 = continuous_dynamics(xx + (dt/2)*k1, uu)
    k3 = continuous_dynamics(xx + (dt/2)*k2, uu)
    k4 = continuous_dynamics(xx + dt*k3, uu)
    
    # x(t_0 + dt) = x(t_0) + (avg slope) * dt
    xx_next = xx + dt * (k1+ 2*k2 + 2*k3 + k4) / 6.0
    
    return xx_next

def continuous_dynamics(xx, uu):
    
    #State and control extraction
    q = xx[0 : 2]
    qdot = xx[2 : 4]
    
    M_val = M_func(*q)
    
    tau_1 = 0.0
    
    RHS_val = RHS_func(*q, *qdot, tau_1, uu[1]).flatten()
    
    qddot = np.linalg.solve(M_val, RHS_val)
    
    x_dot = np.concatenate((qdot, qddot))
    
    return x_dot

def continuous_dynamics_old(xx, uu):
    # Continuous-dynamics of gymnast
    
    xx = xx.squeeze()
    uu = uu.squeeze()
    
    xx_next = np.zeros((ns,))
    
    # Add state to matrices
    M_c = M_num.subs({theta1: xx[0], theta2: xx[1]})
    C_c = C_num.subs({theta1: xx[0], theta2: xx[1], theta1_dot: xx[2], theta2_dot: xx[3]})
    G_c = G_num.subs({theta1: xx[0], theta2: xx[1]})
    
    theta1_dotdot, theta2_dotdot  = sp.symbols('theta1_dotdot theta2_dotdot')
    
    theta_dotdot = sp.Matrix([theta1_dotdot, theta2_dotdot])
    
    eq = sp.Eq( M_c @ theta_dotdot + C_c @ sp.Matrix([xx[2], xx[3]])+ F_num @ sp.Matrix([xx[2], xx[3]]) + G_c, sp.Matrix([0, uu[1]]))
    
    sol = sp.solve(eq, [theta1_dotdot, theta2_dotdot])
    
    xx_next[:,] = np.array([xx[2], xx[3], sol[theta1_dotdot], sol[theta2_dotdot]])
    
    # print(xx_next)
    # print(xx_next.shape)
    
    return xx_next
    
    #formula
    # M_calc = np.array(M_num.subs({theta1: xx[0], theta2: xx[1]}), dtype=float).squeeze()
    # C_calc = np.array(C_num.subs({theta1: xx[0], theta2: xx[1], theta1_dot: xx[2], theta2_dot: xx[3]}), dtype=float).squeeze()
    # G_calc = np.array(G_num.subs({theta1: xx[0], theta2: xx[1]}), dtype=float).squeeze()
    # F_calc = np.array(F_num, dtype=float).squeeze()
    
    # output = M_calc @ xx[2:] + C_calc @ xx[2:] + F_calc @ xx[2:] + G_calc


### Task 2 new ###

def Calculate_A_B_matrixes(x_t, u_t):
    # Fast numerical evaluation
    x_args = x_t.tolist()
    u_args = u_t.tolist()
    
    # Continuous Jacobians
    A_c = np.array(func_A(*x_args, *u_args), dtype=float)
    B_c = np.array(func_B(*x_args, *u_args), dtype=float)
    
    return A_c, B_c
