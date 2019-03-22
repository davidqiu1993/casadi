import casadi as ca


def example_individual_variables():
    opti = ca.Opti()

    x = opti.variable()
    y = opti.variable()

    opti.minimize( (y-x**2)**2 )
    opti.subject_to( x**2+y**2==1 )
    opti.subject_to(       x+y>=1 )

    opti.solver('ipopt')

    sol = opti.solve()

    print(sol.value(x))
    print(sol.value(y))


def example_matrix():
    opti = ca.Opti() # initialize an optimizer instance

    R = ca.DM([ [3.0, 1.0, -0.1],
               [ 1.0, 5.0,  0.5],
               [-0.1, 0.5,  2.0]])

    u = opti.variable(3) # define a variable within the optimizer instance

    opti.minimize( ca.mtimes(u.T, ca.mtimes(R, u)) )
    opti.subject_to( u[0] + u[1] + u[2] >= 5.0 )

    opti.solver('ipopt')

    sol = opti.solve()

    print(sol.value(u))


def main():
    print('----------------------------')
    print('example_individual_variables')
    print('----------------------------')
    example_individual_variables()
    print('')

    print('--------------')
    print('example_matrix')
    print('--------------')
    example_matrix()
    print('')


if __name__ == '__main__':
    main()
