import sympy as sp

# Derivatives
x = sp.Symbol('x')
f = x**2
f1 = sp.sin(x) * sp.exp(x)
derivative = sp.diff(f , x)
derivative_chain_rule = sp.diff(f1 , x)

print("Derivative:\n", derivative)
print("Derivative:\n", derivative_chain_rule)

# Partial Derivatives
x , y = sp.symbols('x y')
f2 = x**2 * y + sp.sin(x * y)
grad_x = sp.diff(f2 , x)
grad_y = sp.diff(f2 , y)
print("w.r.t x= \n", grad_x)
print("w.r.t y= \n", grad_y)
print("")
