"""
AUHTOR: Ryan Grindle

LAST MODIFIED: Mar 18, 2021

PURPOSE: Can I measure how different functional forms are?

NOTES:

TODO:
"""
# import numpy as np
import sympy

x, c0, c1, c2 = sympy.symbols('x c0 c1 c2', real=True)
# f1 = c1*x+c0
# f2 = c2*x**2+c1*x+c0
f1 = x
f2 = x**2+c0
# f3 = 
print(f1)
print(f2)
diff = abs(f1-f2)
print(diff)
ans = sympy.integrate(diff, (x, 0, 1), (c0, 0, 1), (c1, 0, 1), (c2, 0, 1))
print(ans)
