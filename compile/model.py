import os
import time
import numpy as np
import casadi as ca


def f(x, u, dt):
  A = ca.SX([[1, 0, -0,  0],
             [0, 1,  0, -0],
             [0, 0,  1,  0],
             [0, 0,  0,  1]])
  A[0,2] = dt
  A[1,3] = dt

  B = ca.SX([[ 0,  0],
             [ 0,  0],
             [-0,  0],
             [ 0, -0]])
  B[2,0] = dt
  B[3,1] = dt

  x_next = ca.mtimes(A, x) + ca.mtimes(B, u)

  return x_next


def compile_casadi_func(func):
  compile_optimization = 'O1'

  func_name = func.name()
  o_name = func_name + '.so'

  # opts = dict(main=True, with_header=True)
  opts = dict(with_header=True)
  print('Generating %s C-code' % (func_name))
  t1 = time.time()
  c_name = func.generate(opts)
  t2 = time.time()
  print('Generating %s C-code time: %.2f ms' % (func_name, (t2 - t1) * 1e3))
  #compile_flags = '-fPIC -shared -fopenmp -pthread -' + compile_optimization + ' '
  compile_flags = '-fPIC -shared -' + compile_optimization + ' '

  print('Compiling: %s Flags: %s' % (c_name, compile_flags))
  t1 = time.time()
  os.system('gcc ' + compile_flags + c_name + ' -o ' + o_name)
  t2 = time.time()
  print('Compiling %s time: %.2f ms' % (func_name, (t2 - t1) * 1e3))

  func_compiled = ca.external(func_name, './' + o_name)

  return func_compiled



def main():
  x = ca.SX.sym('x', (4, 1))
  u = ca.SX.sym('u', (2, 1))
  dt = ca.SX.sym('dt', (1, 1))

  expr_f = f(x, u, dt)
  expr_f_u = ca.jacobian(expr_f, u)
  eval_f_u = ca.Function('f_u', [x, u, dt], [expr_f_u])

  cmpl_f_u = compile_casadi_func(eval_f_u)

  x_val = ca.DM([1, 1, 1, 1])
  u_val = ca.DM([2, 0.5])
  dt_val = 0.1

  res = None
  T = 100000

  t_run = 0.0
  for t in range(T):
    t1 = time.time()
    res = eval_f_u(x_val, u_val, dt_val)
    t2 = time.time()
    t_run += t2 - t1
  print(np.array(res))
  print('eval_f_u: %.2f ms' % (t_run * 1e3))

  t_run = 0.0
  for t in range(T):
    t1 = time.time()
    res = cmpl_f_u(x_val, u_val, dt_val)
    t2 = time.time()
    t_run += t2 - t1
  print(np.array(res))
  print('cmpl_f_u: %.2f ms' % (t_run * 1e3))


if __name__ == '__main__':
  main()
