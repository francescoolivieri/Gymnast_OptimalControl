import sympy as sp
import numpy as np

#   Symbols 
I1, I2 = sp.symbols('I1 I2')
l1, lc1, l2, lc2 = sp.symbols('l1 lc1 l2 lc2')
m1, m2 = sp.symbols('m1 m2')
theta1, theta2 = sp.symbols('theta1 theta2')
theta1_dot, theta2_dot = sp.symbols('theta1_dot theta2_dot')
g = sp.symbols('g')
f1, f2 = sp.symbols('f1 f2')

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
        
        

dt = 1e-3   # discretization step
ns = 4      # number of states
ni = 1      # number of inputs

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
    
    
    
