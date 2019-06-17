\* Source Pyomo model name=unknown *\

min 
x10:
+1 x1
+4 x2
+3 x3
+4 x4
+2 x5
+5 x6
+4 x7
+4 x8
+7 x9

s.t.

c_e_x11_:
+1 x1
+1 x2
+1 x3
+1 x4
+1 x5
+1 x6
+1 x7
+1 x8
+1 x9
= 5

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
   0 <= x1 <= 1
   0 <= x2 <= 1
   0 <= x3 <= 1
   0 <= x4 <= 1
   0 <= x5 <= 1
   0 <= x6 <= 1
   0 <= x7 <= 1
   0 <= x8 <= 1
   0 <= x9 <= 1
binary
  x1
  x2
  x3
  x4
  x5
  x6
  x7
  x8
  x9
end
