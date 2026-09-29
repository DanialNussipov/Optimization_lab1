import numpy as np
from .models import Problem, Constraint, Sense
from .logger import SimplexLogger

class SimplexSolver:
    def __init__(self, problem: Problem):
        self.prob = problem
        self.num_orig = len(problem.c)
        self.num_cons = len(problem.b)
        self.tableau = None
        self.basis = []
        self.var_names = []
        self.art_vars = []
        
    def _build_tableau(self):
        for i in range(self.num_cons):
            if self.prob.b[i] < 0:
                self.prob.b[i] *= -1
                self.prob.A[i] *= -1
                if self.prob.constraints[i] == Constraint.LEQ:
                    self.prob.constraints[i] = Constraint.GEQ
                elif self.prob.constraints[i] == Constraint.GEQ:
                    self.prob.constraints[i] = Constraint.LEQ

        slack_count = sum(1 for c in self.prob.constraints if c != Constraint.EQ)
        art_count = sum(1 for c in self.prob.constraints if c in (Constraint.GEQ, Constraint.EQ))
        
        total_vars = self.num_orig + slack_count + art_count
        self.tableau = np.zeros((self.num_cons + 1, total_vars + 1), dtype=float)
        
        self.var_names = [f"x{i+1}" for i in range(self.num_orig)]
        curr_var = self.num_orig
        
        for i in range(self.num_cons):
            self.tableau[i, :self.num_orig] = self.prob.A[i]
            self.tableau[i, -1] = self.prob.b[i]
            
            ctype = self.prob.constraints[i]
            if ctype == Constraint.LEQ:
                self.tableau[i, curr_var] = 1.0
                self.var_names.append(f"s{i+1}")
                self.basis.append(curr_var)
                curr_var += 1
            elif ctype == Constraint.GEQ:
                self.tableau[i, curr_var] = -1.0
                self.var_names.append(f"e{i+1}")
                curr_var += 1
                
                self.tableau[i, curr_var] = 1.0
                self.var_names.append(f"a{i+1}")
                self.basis.append(curr_var)
                self.art_vars.append(curr_var)
                curr_var += 1
            elif ctype == Constraint.EQ:
                self.tableau[i, curr_var] = 1.0
                self.var_names.append(f"a{i+1}")
                self.basis.append(curr_var)
                self.art_vars.append(curr_var)
                curr_var += 1

    def _pivot(self, row: int, col: int):
        self.tableau[row] /= self.tableau[row, col]
        factors = self.tableau[:, col].copy()
        factors[row] = 0.0
        self.tableau -= np.outer(factors, self.tableau[row])

    def _run_phase(self, phase: int):
        iteration = 1
        is_max = (self.prob.sense == Sense.MAX)
        
        while True:
            SimplexLogger.log_iteration(phase, iteration, self.tableau, self.basis, self.var_names, is_max)
            
            z_row = self.tableau[-1, :-1]
            if np.all(z_row >= -1e-9):
                return "optimal"
                
            entering = np.argmin(z_row)
            
            col_vals = self.tableau[:-1, entering]
            valid_rows = np.where(col_vals > 1e-9)[0]
            
            if len(valid_rows) == 0:
                return "unbounded"
                
            ratios = self.tableau[valid_rows, -1] / col_vals[valid_rows]
            min_ratio = np.min(ratios)
            
            tied_rows = valid_rows[np.abs(ratios - min_ratio) < 1e-9]
            leaving = min(tied_rows, key=lambda r: self.basis[r])
            
            SimplexLogger.log_pivot(self.var_names[entering], self.var_names[self.basis[leaving]])
            
            self._pivot(leaving, entering)
            self.basis[leaving] = entering
            iteration += 1

    def solve(self):
        self._build_tableau()
        
        # Phase 1
        if self.art_vars:
            print("\n" + "="*40 + "\nСТАРТ ФАЗЫ 1\n" + "="*40)
            self.tableau[-1, :] = 0.0
            for a_idx in self.art_vars:
                r_idx = self.basis.index(a_idx)
                self.tableau[-1, :] -= self.tableau[r_idx, :]
                
            status = self._run_phase(1)
            if status == "unbounded": 
                return "Unbounded", None
            
            # Robust check: if artificial variables remain in the basis with nonzero values, the problem is infeasible
            infeasible = False
            for a_idx in self.art_vars:
                if a_idx in self.basis:
                    r_idx = self.basis.index(a_idx)
                    if abs(self.tableau[r_idx, -1]) > 1e-6:
                        infeasible = True
                        break
            
            if infeasible:
                print("\nResult:Infeasible solution")
                return "Infeasible", None
                
            # Remove artificial-variable columns
            keep_cols = [i for i in range(self.tableau.shape[1]-1) if i not in self.art_vars] + [-1]
            self.tableau = self.tableau[:, keep_cols]
            self.var_names = [self.var_names[i] for i in range(len(self.var_names)) if i not in self.art_vars]
            
            # Adjust basis indices
            for i in range(len(self.basis)):
                shift = sum(1 for a in self.art_vars if a < self.basis[i])
                self.basis[i] -= shift

        # Phase 2
        print("\n" + "="*40 + "\nСТАРТ ФАЗЫ 2\n" + "="*40)
        self.tableau[-1, :] = 0.0
        c_mod = self.prob.c if self.prob.sense == Sense.MAX else -self.prob.c
        self.subtitle_c = c_mod
        self.tableau[-1, :self.num_orig] = -c_mod
        
        for r, b_idx in enumerate(self.basis):
            if b_idx < self.num_orig:
                self.tableau[-1, :] -= self.tableau[-1, b_idx] * self.tableau[r, :]
                
        status = self._run_phase(2)
        
        if status == "unbounded":
            return "Unbounded", None
            
        z_row = self.tableau[-1, :-1]
        non_basic = [i for i in range(len(self.var_names)) if i not in self.basis]
        has_alt = any(abs(z_row[nb]) < 1e-9 and np.any(self.tableau[:-1, nb] > 1e-9) for nb in non_basic)
        
        sol_x = np.zeros(self.num_orig)
        for r, b_idx in enumerate(self.basis):
            if b_idx < self.num_orig:
                sol_x[b_idx] = self.tableau[r, -1]
                
        z_val = self.tableau[-1, -1] if self.prob.sense == Sense.MAX else -self.tableau[-1, -1]
        
        if has_alt:
            print("\nAlternative Optima!")
            
        return "Optimal", {"z": z_val, "x": sol_x}