import pyomo.environ as pm

model = pm.ConcreteModel()

model.x1 = pm.Var(within=pm.NonNegativeReals)
model.x2 = pm.Var(within=pm.NonNegativeReals)
model.x3 = pm.Var(within=pm.NonNegativeReals)

model.cons1 = pm.ConstraintList()
model.cons1.add(model.x1 + model.x2 + model.x3 == 1)
model.cons1.add(model.x1 + model.x2 + model.x3 == 2)

def obj_cost_min(model):
    return 0
model.obj_cost = pm.Objective(rule=obj_cost_min)

opt = pm.SolverFactory("cplex", executable='/Applications/CPLEX_Studio128/cplex/bin/x86-64_osx/cplex')
results = opt.solve(model, tee=True)
print(results['solver'][0]['Termination condition'])
