import numpy as np

class SimplexLogger:
    @staticmethod
    def log_iteration(phase: int, iteration: int, tableau: np.ndarray, basis: list, var_names: list, is_max: bool):
        print(f"\n--- Phase {phase} | Iteration {iteration} ---")

        non_basic = [i for i in range(len(var_names)) if i not in basis]
        
        print("Basic variables:")
        for r, b_idx in enumerate(basis):
            print(f"  {var_names[b_idx]} = {tableau[r, -1]:.4f}")
            
        print("Non-basic variables:", [var_names[i] for i in non_basic])
        
        # Objective function
        z_val = -tableau[-1, -1] if phase == 1 else (tableau[-1, -1] if is_max else -tableau[-1, -1])
        print(f"Current Z value = {z_val:.4f}")

    @staticmethod
    def log_pivot(entering: str, leaving: str):
        print(f"Entering variable: {entering}")
        print(f"Leaving variable: {leaving}")