
## Imports
import numpy as np
from gurobipy import *
from scipy.interpolate import interp1d
from stl import LE, GT


def optimize_inf_gain(traces, primitive, prim_level, prev_rho, pdist, disp = False):

    #### Initialization
    epsilon = 0.001
    M = 100000
    T = traces.length
    if prev_rho is None:
        prev_rho = [10000 for i in traces.labels]

    # interp_data = interp1d(time, data, kind = 'linear')

    min_t = 0
    max_t = max(np.amax(traces.get_sindex(-1), 1))
    min_pi = min(np.amin(traces.get_sindex(primitive.index), 1))
    max_pi = max(np.amax(traces.get_sindex(primitive.index), 1))

    #### Optimization Problem
    m = Model()
    m.setParam('OutputFlag', False)
    # --------------------------------------
    ### Variables
    u = m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY)
    v1 = m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY)
    v2 = m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY)
    v3 = m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY)
    vp = m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY)
    vn = m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY)
    vtp = m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY)
    vtn = m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY)
    w1 = m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY)
    w2 = m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY)
    w3 = m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY)
    z1 = m.addVar(vtype = GRB.BINARY)
    z2 = m.addVar(vtype = GRB.BINARY)
    z3 = m.addVar(vtype = GRB.BINARY)

    vr_pn = [m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY) for i in range(traces.m)]
    v_pn = [m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY) for i in range(traces.m)]
    v_prim = [m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY) for i in range(traces.m)]
    w_pn = [m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY) for i in range(traces.m)]
    w_pn_ar = [m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY) for i in range(traces.m)]
    z_pn = [m.addVar(vtype = GRB.BINARY) for i in range(traces.m)]
    z_pn_ar = [m.addVar(vtype = GRB.BINARY) for i in range(traces.m)]

    vr_tpn = [m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY) for i in range(traces.m)]
    v_tpn = [m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY) for i in range(traces.m)]
    v_tpn_max = [m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY) for i in range(traces.m)]
    w_tpn = [m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY) for i in range(traces.m)]
    w_tpn_max = [m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY) for i in range(traces.m)]
    w_tpn_ar = [m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY) for i in range(traces.m)]
    z_tpn = [m.addVar(vtype = GRB.BINARY) for i in range(traces.m)]
    z_tpn_max = [m.addVar(vtype = GRB.BINARY) for i in range(traces.m)]
    z_tpn_ar = [m.addVar(vtype = GRB.BINARY) for i in range(traces.m)]

    m.update()
    # --------------------------------------
    ### Constraints
    # Initial Constraints
    m.addConstr(w1 <= u)
    m.addConstr(w2 <= u)
    m.addConstr(w3 <= u)
    m.addConstr(w1 <= M * z1)
    m.addConstr(w2 <= M * z2)
    m.addConstr(w3 <= M * z3)
    m.addConstr(w1 >= u - M * (1 - z1))
    m.addConstr(w2 >= u - M * (1 - z2))
    m.addConstr(w3 >= u - M * (1 - z3))
    m.addConstr(u > 0)
    m.addConstr(v1 >= 0)
    m.addConstr(v2 >= 0)
    m.addConstr(v3 >= 0)
    m.addConstr(vp >= 0)
    m.addConstr(vn >= 0)
    m.addConstr(vtp >= 0)
    m.addConstr(vtn >= 0)
    m.addConstr(w1 >= 0)
    m.addConstr(w2 >= 0)
    m.addConstr(w3 >= 0)

    # Equation (30)
    m.addConstr(v1 <= vp)
    m.addConstr(v1 <= vn)
    m.addConstr(v1 >= vp - M * w1)
    m.addConstr(v1 >= vn - M * (u - w1))
    m.addConstr(v2 <= vtp)
    m.addConstr(v2 <= vtn)
    m.addConstr(v2 >= vtp - M * w2)
    m.addConstr(v2 >= vtn - M * (u - w2))
    m.addConstr(v3 <= vp - vtp)
    m.addConstr(v3 <= vn - vtn)
    m.addConstr(v3 >= vp - vtp - M * w3)
    m.addConstr(v3 >= vn - vtn - M * (u - w3))
    m.addConstr(vp + vn == 1)


    # Equation (31)-(34)
    m.addConstr(vp == sum(pdist[i] * vr_pn[i] for i in traces.pos_indices()))
    for i in traces.pos_indices():
        m.addConstr(v_pn[i] <= prev_rho[i] * u)
        m.addConstr(v_pn[i] <= v_prim[i])
        m.addConstr(v_pn[i] >= prev_rho[i] * u - w_pn[i] * M)
        m.addConstr(v_pn[i] >= v_prim[i] - (u - w_pn[i]) * M)
        m.addConstr(vr_pn[i] >= v_pn[i])
        m.addConstr(vr_pn[i] >= -v_pn[i])
        m.addConstr(vr_pn[i] <= v_pn[i] + M * (u - w_pn_ar[i]))
        m.addConstr(vr_pn[i] <= -v_pn[i] + M * w_pn_ar[i])
        m.addConstr(vr_pn[i] >= 0)
        m.addConstr(w_pn[i] <= u)
        m.addConstr(w_pn[i] <= M * z_pn[i])
        m.addConstr(w_pn[i] >= u - M * (1 - z_pn[i]))
        m.addConstr(w_pn[i] >= 0)
        m.addConstr(w_pn_ar[i] <= u)
        m.addConstr(w_pn_ar[i] <= M * z_pn_ar[i])
        m.addConstr(w_pn_ar[i] >= u - M * (1 - z_pn_ar[i]))
        m.addConstr(w_pn_ar[i] >= 0)


    # Equation (35)
    m.addConstr(vn == sum(pdist[i] * vr_pn[i] for i in traces.neg_indices()))
    for i in traces.neg_indices():
        m.addConstr(v_pn[i] <= prev_rho[i] * u)
        m.addConstr(v_pn[i] <= v_prim[i])
        m.addConstr(v_pn[i] >= prev_rho[i] * u - w_pn[i] * M)
        m.addConstr(v_pn[i] >= v_prim[i] - (u - w_pn[i]) * M)
        m.addConstr(vr_pn[i] >= v_pn[i])
        m.addConstr(vr_pn[i] >= -v_pn[i])
        m.addConstr(vr_pn[i] <= v_pn[i] + M * (u - w_pn_ar[i]))
        m.addConstr(vr_pn[i] <= -v_pn[i] + M * w_pn_ar[i])
        m.addConstr(vr_pn[i] >= 0)
        m.addConstr(w_pn[i] <= u)
        m.addConstr(w_pn[i] <= M * z_pn[i])
        m.addConstr(w_pn[i] >= u - M * (1 - z_pn[i]))
        m.addConstr(w_pn[i] >= 0)
        m.addConstr(w_pn_ar[i] <= u)
        m.addConstr(w_pn_ar[i] <= M * z_pn_ar[i])
        m.addConstr(w_pn_ar[i] >= u - M * (1 - z_pn_ar[i]))
        m.addConstr(w_pn_ar[i] >= 0)


    # Equation (38)-(43)
    m.addConstr(vtp == sum(pdist[i] * vr_tpn[i] for i in traces.pos_indices()))
    for i in traces.pos_indices():
        m.addConstr(v_tpn_max[i] >= v_prim[i])
        m.addConstr(v_tpn_max[i] >= epsilon * u)
        m.addConstr(v_tpn_max[i] <= v_prim[i] + M * (u - w_tpn_max[i]))
        m.addConstr(v_tpn_max[i] <= epsilon * u + M * w_tpn_max[i])
        m.addConstr(v_tpn[i] <= prev_rho[i] * u)
        m.addConstr(v_tpn[i] <= v_tpn_max[i])
        m.addConstr(v_tpn[i] >= prev_rho[i] * u - M * w_tpn[i])
        m.addConstr(v_tpn[i] >= v_tpn_max[i] - M * (u - w_tpn[i]))
        m.addConstr(vr_tpn[i] >= v_tpn[i])
        m.addConstr(vr_tpn[i] >= -v_tpn[i])
        m.addConstr(vr_tpn[i] <= v_tpn[i] + M * (u - w_tpn_ar[i]))
        m.addConstr(vr_tpn[i] <= -v_tpn[i] + M * w_tpn_ar[i])
        m.addConstr(vr_tpn[i] >= 0)
        m.addConstr(w_tpn_max[i] <= u)
        m.addConstr(w_tpn_max[i] <= M * z_tpn_max[i])
        m.addConstr(w_tpn_max[i] >= u - M * (1 - z_tpn_max[i]))
        m.addConstr(w_tpn_max[i] >= 0)
        m.addConstr(w_tpn[i] <= u)
        m.addConstr(w_tpn[i] <= M * z_tpn[i])
        m.addConstr(w_tpn[i] >= u - M * (1 - z_tpn[i]))
        m.addConstr(w_tpn[i] >= 0)
        m.addConstr(w_tpn_ar[i] <= u)
        m.addConstr(w_tpn_ar[i] <= M * z_tpn_ar[i])
        m.addConstr(w_tpn_ar[i] >= u - M * (1 - z_tpn_ar[i]))
        m.addConstr(w_tpn_ar[i] >= 0)

    # Same Equations as (38)-(43) for Rtn
    m.addConstr(vtn == sum(pdist[i] * vr_tpn[i] for i in traces.neg_indices()))
    for i in traces.neg_indices():
        m.addConstr(v_tpn_max[i] >= v_prim[i])
        m.addConstr(v_tpn_max[i] >= epsilon * u)
        m.addConstr(v_tpn_max[i] <= v_prim[i] + M * (u - w_tpn_max[i]))
        m.addConstr(v_tpn_max[i] <= epsilon * u + M * w_tpn_max[i])
        m.addConstr(v_tpn[i] <= prev_rho[i] * u)
        m.addConstr(v_tpn[i] <= v_tpn_max[i])
        m.addConstr(v_tpn[i] >= prev_rho[i] * u - M * w_tpn[i])
        m.addConstr(v_tpn[i] >= v_tpn_max[i] - M * (u - w_tpn[i]))
        m.addConstr(vr_tpn[i] >= v_tpn[i])
        m.addConstr(vr_tpn[i] >= -v_tpn[i])
        m.addConstr(vr_tpn[i] <= v_tpn[i] + M * (u - w_tpn_ar[i]))
        m.addConstr(vr_tpn[i] <= -v_tpn[i] + M * w_tpn_ar[i])
        m.addConstr(vr_tpn[i] >= 0)
        m.addConstr(w_tpn_max[i] <= u)
        m.addConstr(w_tpn_max[i] <= M * z_tpn_max[i])
        m.addConstr(w_tpn_max[i] >= u - M * (1 - z_tpn_max[i]))
        m.addConstr(w_tpn_max[i] >= 0)
        m.addConstr(w_tpn[i] <= u)
        m.addConstr(w_tpn[i] <= M * z_tpn[i])
        m.addConstr(w_tpn[i] >= u - M * (1 - z_tpn[i]))
        m.addConstr(w_tpn[i] >= 0)
        m.addConstr(w_tpn_ar[i] <= u)
        m.addConstr(w_tpn_ar[i] <= M * z_tpn_ar[i])
        m.addConstr(w_tpn_ar[i] >= u - M * (1 - z_tpn_ar[i]))
        m.addConstr(w_tpn_ar[i] >= 0)

    # Adding the constraints of the primitive robustness
    if prim_level:
        expr = primitive._op
        index = primitive.index
        op = primitive.args[0].args[0].op

        pi = (min_pi + max_pi)/7
        v_pi = m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY)
        v_prim_max = [[m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY) for t in range(T)] for i in range(traces.m)]
        w_prim = [[m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY) for t in range(T)] for i in range(traces.m)]
        w_prim_max = [[m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY) for t in range(T)] for i in range(traces.m)]
        w_id_t = [m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY) for t in range(T)]
        w_id_t_inc = [m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY) for t in range(T)]
        w_id_t_dec = [m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY) for t in range(T)]
        z_prim = [[m.addVar(vtype = GRB.BINARY) for t in range(T)] for i in range(traces.m)]
        z_prim_max = [[m.addVar(vtype = GRB.BINARY) for t in range(T)] for i in range(traces.m)]
        id_t = [m.addVar(vtype = GRB.BINARY) for t in range(T)]
        max_id_t = m.addVar(vtype = GRB.BINARY)
        id_t_inc = [m.addVar(vtype = GRB.BINARY) for t in range(T)]
        id_t_dec = [m.addVar(vtype = GRB.BINARY) for t in range(T)]
        m.update()


        m.addConstr(sum(id_t) >= 1)
        m.addConstr(v_pi <= u * max_pi)
        m.addConstr(v_pi >= u * min_pi)
        for t in range(T):
            m.addConstr(w_id_t[t] <= u)
            m.addConstr(w_id_t[t] <= M * id_t[t])
            m.addConstr(w_id_t[t] >= u - M * (1 - id_t[t]))
            m.addConstr(w_id_t[t] >= 0)
            m.addConstr(w_id_t_inc[t] <= u)
            m.addConstr(w_id_t_inc[t] <= M * id_t_inc[t])
            m.addConstr(w_id_t_inc[t] >= u - M * (1 - id_t_inc[t]))
            m.addConstr(w_id_t_inc[t] >= 0)
            m.addConstr(w_id_t_dec[t] <= u)
            m.addConstr(w_id_t_dec[t] <= M * id_t_dec[t])
            m.addConstr(w_id_t_dec[t] >= u - M * (1 - id_t_dec[t]))
            m.addConstr(w_id_t_dec[t] >= 0)
            for i in range(traces.m):
                m.addConstr(w_prim[i][t] <= u)
                m.addConstr(w_prim[i][t] <= M * z_prim[i][t])
                m.addConstr(w_prim[i][t] >= u - M * (1 - z_prim[i][t]))
                m.addConstr(w_prim[i][t] >= 0)
                m.addConstr(w_prim_max[i][t] <= u)
                m.addConstr(w_prim_max[i][t] <= M * z_prim_max[i][t])
                m.addConstr(w_prim_max[i][t] >= u - M * (1 - z_prim_max[i][t]))
                m.addConstr(w_prim_max[i][t] >= 0)

        for i in range(traces.m):
            m.addConstr(sum(w_prim[i][t] for t in range(T)) == u)
            # m.addQConstr(r_prim[i] == sum(z_prim[i][t] * r_prim_max[i][t] for t in range(T)))
            for t in range(T):
                # always operator
                if expr == 5:
                    m.addConstr(v_prim[i] <= v_prim_max[i][t])
                    m.addConstr(v_prim[i] >= v_prim_max[i][t] - w_prim[i][t] * M)
                # eventually operator
                else:
                    m.addConstr(v_prim[i] >= v_prim_max[i][t])
                    m.addConstr(v_prim[i] <= v_prim_max[i][t] + M * (u - w_prim[i][t]))

                m.addConstr(v_prim_max[i][t] >= M * (u - 2 * w_id_t[t]))
                m.addConstr(v_prim_max[i][t] <=  M * (u - 2 * w_id_t[t]) + M * w_prim_max[i][t])
                if op == LE:
                    m.addConstr(v_prim_max[i][t] >= v_pi - u * traces.signals[i][index][t])
                    m.addQConstr(v_prim_max[i][t] <= v_pi - u * traces.signals[i][index][t] + M * (u - w_prim_max[i][t]))
                else:
                    m.addConstr(v_prim_max[i][t] >= u * traces.signals[i][index][t] - v_pi)
                    m.addQConstr(v_prim_max[i][t] <= u * traces.signals[i][index][t] - v_pi + M * (u - w_prim_max[i][t]))

        for t in range(T):
            m.addConstr(id_t[t] <= id_t_inc[t])
            m.addConstr(id_t[t] <= id_t_dec[t])

        for t in range(T-1):
            m.addConstr(id_t_inc[t] <= id_t_inc[t+1])
            m.addConstr(id_t_dec[t] >= id_t_dec[t+1])


    # Second-level primitives
    else:
        t1 = m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY)
        t2 = m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY)
        t3 = m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY)
        pi = m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY)
        out_expr = primitive._op
        in_expr = primitive.args[0]._op
        index = primitive.args[0].args[0].args[0].index
        op = primitive.args[0].args[0].args[0].op

        # always(eventually) operator
        # if expr == 5:

        # eventually(always) operator
        # else:

    # --------------------------------------
    ### Objective Function
    m.setObjective(v1 - v2 - v3, GRB.MAXIMIZE)
    m.update()

    m.optimize()
    print(m.status)
    # m.computeIIS()
    # m.write("model.ilp")
    # m.write("model.mps")

    if m.status == 2:
        primitive.pi = v_pi.X/u.X
        print(primitive.pi)
        print(id_t_inc)
        print(id_t_dec)
        for t in range(T-1):
            if (id_t_inc[t].X < 0.5) and (id_t_inc[t+1].X >= 0.5):
            # if not id_t_inc[t].X and id_t_inc[t+1].X:
        # if not w_id_t_inc[t].X/u.X and w_id_t_inc[t+1].X/u.X:
                t0 = t+1
                print(t0)
                # break
        for t in range(T-1):
            if (id_t_dec[t].X >= 0.5) and (id_t_dec[t+1].X < 0.5):
            # if id_t_dec[t].X and not id_t_dec[t+1].X:
        # if w_id_t_dec[t].X/u.X and not w_id_t_dec[t+1].X/u.X:
                t1 = t
                print(t1)
                # break
        # primitive.t0 = t0
        # primitive.t1 = t1

        return primitive, m.getObjective().getValue()
    else:
        return None, None
