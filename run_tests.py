from simplex_lib.logger import SimplexLogger
from simplex_lib.models import Problem, Constraint, Sense
from simplex_lib.solver import SimplexSolver
import numpy as np

def run_problem_1():
    print("\n" + "#"*50 + "\nPROBLEM 1: Maximization\n" + "#"*50)
    p = Problem(sense=Sense.MAX, c=np.array([5, 4]))
    p.add_constraint([6, 4], Constraint.LEQ, 24)
    p.add_constraint([1, 2], Constraint.LEQ, 6)
    p.add_constraint([-1, 1], Constraint.LEQ, 1)
    p.add_constraint([0, 1], Constraint.LEQ, 2)
    
    solver = SimplexSolver(p)
    status, res = solver.solve()
    print(f"\nFINAL: {status}", res)

def run_problem_2():
    print("\n" + "#"*50 + "\nPROBLEM 2: Minimization, all types of constraints\n" + "#"*50)
    p = Problem(sense=Sense.MIN, c=np.array([4, 1]))
    p.add_constraint([3, 1], Constraint.EQ, 3)
    p.add_constraint([4, 3], Constraint.GEQ, 6)
    p.add_constraint([1, 2], Constraint.LEQ, 4)
    
    solver = SimplexSolver(p)
    status, res = solver.solve()
    print(f"\nFINAL: {status}", res)

def run_problem_3():
    print("\n" + "#"*50 + "\nPROBLEM 3: Maximization, 6 variables, 5 constraints\n" + "#"*50)
    # Maximization Z = 9x1 + 5x2 + 10x3 + 7x4 + 5x5 + 12x6
    p = Problem(sense=Sense.MAX, c=np.array([9, 5, 10, 7, 5, 12]))
    p.add_constraint([0, 0, 3, 2, 1, 2], Constraint.LEQ, 45)
    p.add_constraint([4, 4, 0, 0, 2, 4], Constraint.LEQ, 20)
    p.add_constraint([3, 3, 0, 1, 3, 3], Constraint.LEQ, 24)
    p.add_constraint([1, 3, 1, 1, 2, 2], Constraint.GEQ, 12)
    p.add_constraint([0, 4, 1, 4, 0, 2], Constraint.EQ, 10)
    
    solver = SimplexSolver(p)
    status, res = solver.solve()
    print(f"\nFINAL: {status}", res)

def run_problem_4():
    print("\n" + "#"*50 + "\nPROBLEM 4: Minimization, 6 variables, 5 constraints\n" + "#"*50)
    # Minimization z = 3x1 + 5x2 + 8x3 + 6x4 + 6x5 + 2x6
    p = Problem(sense=Sense.MIN, c=np.array([3, 5, 8, 6, 6, 2]))
    p.add_constraint([3, 3, 0, 1, 2, 2], Constraint.GEQ, 18)
    p.add_constraint([0, 0, 0, 1, 1, 1], Constraint.GEQ, 20)
    p.add_constraint([3, 0, 3, 1, 0, 1], Constraint.EQ, 24)
    p.add_constraint([2, 2, 2, 2, 1, 1], Constraint.LEQ, 40)
    p.add_constraint([0, 2, 3, 0, 1, 0], Constraint.GEQ, 8)
    
    solver = SimplexSolver(p)
    status, res = solver.solve()
    print(f"\nFINAL: {status}", res)

def run_problem_5():
    print("\n" + "#"*50 + "\nPROBLEM 5: Infeasible solution\n" + "#"*50)
    # Maximization z = 3x1 + 2x2, constraints: 2x1 + x2 <= 2, 3x1 + 4x2 >= 12
    p = Problem(sense=Sense.MAX, c=np.array([3, 2]))
    p.add_constraint([2, 1], Constraint.LEQ, 2)
    p.add_constraint([3, 4], Constraint.GEQ, 12)
    
    solver = SimplexSolver(p)
    status, res = solver.solve()
    print(f"\nFINAL: {status}", res)

def run_problem_6():
    print("\n" + "#"*50 + "\nPROBLEM 6: Unbounded solution\n" + "#"*50)
    # Maximization z = x1 + 2x2 + 3x3[cite: 1]
    p = Problem(sense=Sense.MAX, c=np.array([1, 2, 3]))
    p.add_constraint([-1, 1, 1], Constraint.LEQ, 6)
    p.add_constraint([1, 2, -1], Constraint.LEQ, 4)
    
    solver = SimplexSolver(p)
    status, res = solver.solve()
    print(f"\nFINAL: {status}", res)

def run_problem_7():
    print("\n" + "#"*50 + "\nPROBLEM 7: Alternative optima\n" + "#"*50)
    # Maximization z = 3x1 + 2x2 + 6x3[cite: 1]
    p = Problem(sense=Sense.MAX, c=np.array([3, 2, 6]))
    p.add_constraint([2, 1, 3], Constraint.LEQ, 12)
    p.add_constraint([1, 1, 1], Constraint.LEQ, 6)
    
    solver = SimplexSolver(p)
    status, res = solver.solve()
    print(f"\nFINAL: {status}", res)

def run_problem_8():
    print("\n" + "#"*50 + "\nPROBLEM 8: Degeneracy\n" + "#"*50)
    p = Problem(sense=Sense.MAX, c=np.array([3, 1, 1]))
    p.add_constraint([1, 1, 0], Constraint.LEQ, 4)
    p.add_constraint([1, 0, 1], Constraint.LEQ, 4)
    
    solver = SimplexSolver(p)
    status, res = solver.solve()
    print(f"\nFINAL: {status}", res)


if __name__ == "__main__":
    run_problem_1()
    run_problem_2()
    run_problem_3()
    run_problem_4()
    run_problem_5() # Expected status: "Infeasible"
    run_problem_6() # Expected status: "Unbounded"
    run_problem_7()
    run_problem_8()

