This directory contains code for doing inference of STL formulas from a set of trajectories. This code was provided by researchers working on a project by the name dt-tli-non-inc-batch and was not authored as part of the HAL-Suite project. 

The functionality taken from the orginial dt-tli branch is encapsulated in the following functions. The make_stl_primitives_1 and learn_formula functions used directly by hal_suite, the rest of the functions are called by those. 

(\hal_suite)
- _generate_formula
    (\dt-tli)
    - make_stl_primitives1
    - learn_formula
        - boosted_trees
        - best_prim (stl_inf.py)
            - run_pso_optimization (stl_inf.py)
                - get_bounds (pso_test.py)
                - PSO (pso.py)
                    - initialize_swarm (pso.py)
                        - Particle (combined_pso.py and pso.py)
                    - pso_costFunc (pso.py)
                        - compute_robustness.py (pso.py)
                - optimize_swarm (pso.py and combined_pso.py have this functionalty duplicated)
                    - evaluate (pso.py and combined_pso.py)
                    - update_velocity  (pso.py and combined_pso.py)
                    - update_position  (pso.py and combined_pso.py)

            - set_stl1_pars (stl_prim.py)
            - set_stl2_pars (stl_prim.py)
            - compute_robustness (pso.py)
            - reverse_primitive (stl_prim.py)
            - combine_primitives (stl_inf.py)
                - best_combined_prim (stl_inf.py)
                    - run_combined_pso (combined_pso_test.py)
                        - get_indices (combined_pso_test.py)
                        - get_bounds  (combined_pso_test.py)  
                    - set_combined_stl_pars (stl_prim.py)
                    - compute_combined_robustness (combined_pso.py)
            - 
        - pruned_tree (stl_inf.py)
        - normal_tree (stl_inf.py)
        - bdt_evaluation (non_inc.py)

