
import numpy as np
from gurobipy import *
from scipy.interpolate import interp1d
from stl import LE, GT


def optimize_inf_gain(traces, primitive, prim_level, prev_rho, pdist, disp = False):

    #### Initialization
    epsilon = 0.1
    M = 1000
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
    u = m.addVar(lb = 0, ub = GRB.INFINITY)
    w1 = m.addVar(lb = 0, ub = GRB.INFINITY)
    w2 = m.addVar(lb = 0, ub = GRB.INFINITY)
    w3 = m.addVar(lb = 0, ub = GRB.INFINITY)
    x1 = m.addVar(lb = 0, ub = GRB.INFINITY)
    x2 = m.addVar(lb = 0, ub = GRB.INFINITY)
    x3 = m.addVar(lb = 0, ub = GRB.INFINITY)
    x4 = m.addVar(lb = 0, ub = GRB.INFINITY)
    z1 = m.addVar(vtype = GRB.BINARY)
    z2 = m.addVar(vtype = GRB.BINARY)
    z3 = m.addVar(vtype = GRB.BINARY)
    Rp = m.addVar(lb = 0, ub = GRB.INFINITY)
    Rn = m.addVar(lb = 0, ub = GRB.INFINITY)
    Rtp = m.addVar(lb = 0, ub = GRB.INFINITY)
    Rtn = m.addVar(lb = 0, ub = GRB.INFINITY)

    ar_pn = [m.addVar(lb = 0, ub = GRB.INFINITY) for i in range(traces.m)]
    r_pn = [m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY) for i in range(traces.m)]
    r_prim = [m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY) for i in range(traces.m)]
    z_pn = [m.addVar(vtype = GRB.BINARY) for i in range(traces.m)]
    z_pn_ar = [m.addVar(vtype = GRB.BINARY) for i in range(traces.m)]

    ar_tpn = [m.addVar(lb = 0, ub = GRB.INFINITY) for i in range(traces.m)]
    r_tpn = [m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY) for i in range(traces.m)]
    r_tpn_max = [m.addVar(lb = 0, ub = GRB.INFINITY) for i in range(traces.m)]
    z_tpn = [m.addVar(vtype = GRB.BINARY) for i in range(traces.m)]
    z_tpn_max = [m.addVar(vtype = GRB.BINARY) for i in range(traces.m)]
    z_tpn_ar = [m.addVar(vtype = GRB.BINARY) for i in range(traces.m)]

    m.update()
    # --------------------------------------
    ### Constraints
    m.addQConstr(w1 == x1 * u)
    m.addQConstr(w2 == x2 * u)
    m.addQConstr(w3 == x3 * u)
    m.addQConstr(u * x4 == 1)
    m.addConstr(u >= 0)

    # equation (30)
    m.addConstr(x1 <= Rp)
    m.addConstr(x1 <= Rn)
    m.addQConstr(x1 == z1 * Rp + (1 - z1) * Rn)
    m.addConstr(x2 <= Rtp)
    m.addConstr(x2 <= Rtn)
    m.addQConstr(x2 == z2 * Rtp + (1 - z2) * Rtn)
    m.addConstr(x3 <= (Rp - Rtp))
    m.addConstr(x3 <= (Rn - Rtn))
    m.addQConstr(x3 == z3 * (Rp - Rtp) + (1 - z3) * (Rn - Rtn))
    m.addConstr(x4 == Rp + Rn)

    # equation (33)
    m.addConstr(Rp == sum(pdist[i] * ar_pn[i] for i in traces.pos_indices()))
    # equation (34)
    for i in traces.pos_indices():
        m.addConstr(r_pn[i] <= prev_rho[i])
        m.addConstr(r_pn[i] <= r_prim[i])
        m.addQConstr(r_pn[i] == z_pn[i] * prev_rho[i] + (1 - z_pn[i]) * r_prim[i])
        m.addConstr(ar_pn[i] >= r_pn[i])
        m.addConstr(ar_pn[i] <= -r_pn[i])
        m.addQConstr(ar_pn[i] == (2 * z_pn_ar[i] - 1) * r_pn[i])

    # equation (37)
    m.addConstr(Rn == sum(pdist[i] * ar_pn[i] for i in traces.neg_indices()))
    # similar to equation (34) for negative indices
    for i in traces.neg_indices():
        m.addConstr(r_pn[i] <= prev_rho[i])
        m.addConstr(r_pn[i] <= r_prim[i])
        m.addQConstr(r_pn[i] == z_pn[i] * prev_rho[i] + (1 - z_pn[i]) * r_prim[i])
        m.addConstr(ar_pn[i] >= r_pn[i])
        m.addConstr(ar_pn[i] <= -r_pn[i])
        m.addQConstr(ar_pn[i] == (2 * z_pn_ar[i] - 1) * r_pn[i])

    # equation (42)
    m.addConstr(Rtp == sum(pdist[i] * ar_tpn[i] for i in traces.pos_indices()))
    # equation (43)
    for i in traces.pos_indices():
        m.addConstr(r_tpn_max[i] >= r_prim[i])
        m.addConstr(r_tpn_max[i] >= epsilon)
        m.addQConstr(r_tpn_max[i] == z_tpn_max[i] * r_prim[i] + (1 - z_tpn_max[i]) * epsilon)
        m.addConstr(r_tpn[i] <= prev_rho[i])
        m.addConstr(r_tpn[i] <= r_tpn_max[i])
        m.addQConstr(r_tpn[i] == z_tpn[i] * prev_rho[i] + (1 - z_tpn[i]) * r_tpn_max[i])
        m.addConstr(ar_tpn[i] >= r_tpn[i])
        m.addConstr(ar_tpn[i] >= -r_tpn[i])
        m.addQConstr(ar_tpn[i] == (2 * z_tpn_ar[i] - 1) * r_tpn[i])

    # Same equations of (42)-(43) for Rtn
    m.addConstr(Rtn == sum(pdist[i] * ar_tpn[i] for i in traces.neg_indices()))
    for i in traces.neg_indices():
        m.addConstr(r_tpn_max[i] >= r_prim[i])
        m.addConstr(r_tpn_max[i] >= epsilon)
        m.addQConstr(r_tpn_max[i] == z_tpn_max[i] * r_prim[i] + (1 - z_tpn_max[i]) * epsilon)
        m.addConstr(r_tpn[i] <= prev_rho[i])
        m.addConstr(r_tpn[i] <= r_tpn_max[i])
        m.addQConstr(r_tpn[i] == z_tpn[i] * prev_rho[i] + (1 - z_tpn[i]) * r_tpn_max[i])
        m.addConstr(ar_tpn[i] >= r_tpn[i])
        m.addConstr(ar_tpn[i] >= -r_tpn[i])
        m.addQConstr(ar_tpn[i] == (2 * z_tpn_ar[i] - 1) * r_tpn[i])

    # Adding the constraints for the primitive's robustness
    # m = prim_rho(m, traces, primitive, prim_level)

    if prim_level:
        expr = primitive._op
        index = primitive.index
        op = primitive.args[0].args[0].op

        pi = m.addVar(lb = min_pi, ub = max_pi)
        r_prim_max = [[m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY) for t in range(T)] for i in range(traces.m)]
        z_prim = [[m.addVar(vtype = GRB.BINARY) for t in range(T)] for i in range(traces.m)]
        z_prim_max = [[m.addVar(vtype = GRB.BINARY) for t in range(T)] for i in range(traces.m)]
        id_t = [m.addVar(vtype = GRB.BINARY) for t in range(T)]
        id_t_inc = [m.addVar(vtype = GRB.BINARY) for t in range(T)]
        id_t_dec = [m.addVar(vtype = GRB.BINARY) for t in range(T)]
        m.update()

        for i in range(traces.m):
            m.addConstr(sum(z_prim_max[i][t] for t in range(T)) == 1)
            m.addQConstr(r_prim[i] == sum(z_prim[i][t] * r_prim_max[i][t] for t in range(T)))
            for t in range(T):
                # m.addQConstr(r_prim[i] == z_prim[i][t] * r_prim_max[i][t])
                # always operator
                if expr == 5:
                    m.addConstr(r_prim[i] <= r_prim_max[i][t])
                # eventually operator
                else:
                    m.addConstr(r_prim[i] >= r_prim_max[i][t])

                m.addConstr(r_prim_max[i][t] >= M * (1 - 2 * id_t[t]))
                if op == LE:
                    m.addConstr(r_prim_max[i][t] >= pi - traces.signals[i][index][t])
                    m.addQConstr(r_prim_max[i][t] == z_prim_max[i][t] * (pi - traces.signals[i][index][t]) + (1 - z_prim_max[i][t]) * M * (1 - 2 * id_t[t]))
                else:
                    m.addConstr(r_prim_max[i][t] >= traces.signals[i][index][t] - pi)
                    m.addQConstr(r_prim_max[i][t] == z_prim_max[i][t] * (traces.signals[i][index][t] - pi) + (1 - z_prim_max[i][t]) * M * (1 - 2 * id_t[t]))

        for t in range(T):
            m.addConstr(id_t[t] <= id_t_inc[t])
            m.addConstr(id_t[t] <= id_t_dec[t])
        for t in range(T-1):
            m.addConstr(id_t_inc[t] <= id_t_inc[t+1])
            m.addConstr(id_t_dec[t] >= id_t_dec[t+1])
        m.addConstr(sum(id_t[t] for t in range(T)) == 1)




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



    m.update()


    # --------------------------------------
    # Objective Function
    m.setObjective(w1 - w2 - w3, GRB.MAXIMIZE)
    m.update()

    m.optimize()
    print(m.status)
    m.computeIIS()
    m.write("model.ilp")
    m.write("model.mps")

    primitive.pi = pi.X
    for t in range(T-1):
        if not id_t_inc[t].X and id_t_inc[t+1].X:
            t0 = t+1
            break
    for t in range(T-1):
        if id_t_dec[t].X and not id_t_dec[t+1].X:
            t1 = t
            break
    primitive.t0 = t0
    primitive.t1 = t1

    return primitive, m.getObjective().getValue()
