/*                            2D Wave equation                                */
/*   Version 5: Has a fourth-order Runge-Kutta method applied. This version   */
/*              attempts to find the convergence rate and afterward will      */
/*              implement a vector valued output problem.                     */


// Must create triangulation and a grid
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
// DoF handler and finite elements
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
// Want to show how grid or data looks like
#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/utilities.h>
// Want to use the dynamic sparsity pattern and sparsity pattern
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
// Need to create Mass and Laplace matrices
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
// Data output for visualisation
#include <deal.II/numerics/data_out.h>
// Precondition and solver
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>


#include <fstream>
#include <iostream>
#include <cmath>

using namespace dealii;

/*----------------------------------------------------------------------------*/
/*--------------------------- Forcing Function -------------------------------*/
/*----------------------------------------------------------------------------*/
template <int dim>
class ForcingFunction : public Function<dim>
{
  public:
    ForcingFunction(const unsigned int n_components = 1, const double time = 0.)
      : Function<dim>(n_components, time) {}
    virtual
    double value(const Point<dim> &p, unsigned int component = 0) const override
    {
      double t = this->get_time();
      (void)component;
      return std::sin(p[0]) * std::sin(p[1]) * std::sin(t);
    }
};

/*----------------------------------------------------------------------------*/
/*----------------------------- Exact Function -------------------------------*/
/*----------------------------------------------------------------------------*/
template <int dim>
class ExactSolution : public Function<dim>
{
  public:
    ExactSolution(const unsigned int n_components = 1, const double time = 0.)
      : Function<dim>(n_components, time) {}
    virtual
    double value(const Point<dim> &p, unsigned int component = 0) const override
    {
      // double t = this->get_time();
      (void)p;
      (void)component;
      return 0.0;
    }
};

/*----------------------------------------------------------------------------*/
/*------------------------ Exact Function Derivative -------------------------*/
/*----------------------------------------------------------------------------*/
template <int dim>
class ExactSolutionDerivative : public Function<dim>
{
  public:
    ExactSolutionDerivative(const unsigned int n_components = 1, const double time = 0.)
      : Function<dim>(n_components, time) {}
    virtual
    double value(const Point<dim> &p, unsigned int component = 0) const override
    {
      // double t = this->get_time();
      (void)component;
      (void)p;
      // return std::sin(p[0]) * std::sin(p[1]) * std::cos(t);
      return std::sin(p[0]) * std::sin(p[1]);
    }
};

/*----------------------------------------------------------------------------*/
/*-------------------------------- Boundary ----------------------------------*/
/*----------------------------------------------------------------------------*/
template <int dim>
class Boundary : public Function<dim>
{
  public:
    Boundary(const unsigned int n_components = 1, const double time = 0.)
      : Function<dim>(n_components, time) {}
    virtual
    double value(const Point<dim> &p, unsigned int component = 0) const override
    {
      (void)component;
      (void)p;
      return 0.0;
    }
};

/*----------------------------------------------------------------------------*/
/*--------------------------- Boundary Derivative ----------------------------*/
/*----------------------------------------------------------------------------*/
template <int dim>
class BoundaryDerivative : public Function<dim>
{
  public:
    BoundaryDerivative(const unsigned int n_components = 1, const double time = 0.)
      : Function<dim>(n_components, time) {}
    virtual
    double value(const Point<dim> &p, unsigned int component = 0) const override
    {
      (void)component;
      (void)p;
      return 0.0;
    }
};

/*----------------------------------------------------------------------------*/
/*----------------------------- Class Template -------------------------------*/
/*----------------------------------------------------------------------------*/
template <int dim>
class ElasticWaveEquation
{
public:
  ElasticWaveEquation();
  void run();
private:
  void create_grid();
  void initialise_system();
  void assemble_nth_iteration();
  void graph(const unsigned int time_step);

  Triangulation<dim>        triangulation;
  DoFHandler<dim>           dof_handler;
  FESystem<dim>             fe;
  const unsigned int        n_refinements;

  DynamicSparsityPattern    dynamic_sparsity_pattern;
  SparsityPattern           sparsity_pattern;

  AffineConstraints<double> constraints;

  Vector<double>            Solution_u;
  Vector<double>            Solution_v;

  SparseMatrix<double>      MassMatrix;
  SparseMatrix<double>      LaplaceMatrix;
  Vector<double>            F_n;
  Vector<double>            F_nh;
  Vector<double>            F_np;

  SparseMatrix<double>      LHS_Matrix_u;
  SparseMatrix<double>      LHS_Matrix_v;
  SparseMatrix<double>      LHS_Matrix_u_temp;
  SparseMatrix<double>      LHS_Matrix_v_temp;

  Vector<double>            R_u;
  Vector<double>            R_v;
  Vector<double>            R_u_temp1;
  Vector<double>            R_v_temp1;
  Vector<double>            R_u_temp2;
  Vector<double>            R_v_temp2;
  Vector<double>            R_u_temp3;
  Vector<double>            W_u;
  Vector<double>            W_v;
  Vector<double>            W_v_temp;
  Vector<double>            U_temp;
  Vector<double>            V_temp;
  Vector<double>            RHS_u;
  Vector<double>            RHS_v;

  const double              cfl;
  const double              csquared;
  const double              h;
  const unsigned int        timesteps;
};

/*------------------------------ Constructor ---------------------------------*/
template <int dim>
ElasticWaveEquation<dim>::ElasticWaveEquation()
: dof_handler(triangulation)
, fe(FE_Q<dim>(2))
, n_refinements(5)
, cfl(0.90)
, csquared(1.0)
, h(cfl/(csquared*std::pow(2.0,n_refinements+1.0)))   // k < h/c = 2^(n_refine + 1)/c
// , k(1.0/64.0)
, timesteps(250)
{}

/*----------------------------- Creates Grid ---------------------------------*/
// This creates the grid. We'll use a circle for now.
template <int dim>
void ElasticWaveEquation<dim>::create_grid()
{
  GridGenerator::hyper_cube(triangulation,
                            0.,
                            2.0*numbers::PI,
                            false);
  triangulation.refine_global(n_refinements);
}

/*---------------------------- Sets up System --------------------------------*/
// Set up the system to be solved by locating where the degrees of freedom are
// in fe. Then creates the matrix using a sparse pattern.
template <int dim>
void ElasticWaveEquation<dim>::initialise_system()
{
  // Need this to know which nodes have how many degrees of freedom
  dof_handler.distribute_dofs(fe);
  // Using the above information, a system matrix can be produced of the right size.
  dynamic_sparsity_pattern.reinit(dof_handler.n_dofs(),dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern);
  sparsity_pattern.copy_from(dynamic_sparsity_pattern);

  constraints.close();

  // Initialises solution for u and v
  Solution_u.reinit(dof_handler.n_dofs());
  Solution_v.reinit(dof_handler.n_dofs());
  ExactSolution<dim> ES;
  ES.set_time(0.0);
  VectorTools::project(dof_handler,
                       constraints,
                       QGauss<dim>(fe.degree + 1),
                       ES,
                       Solution_u);
  ExactSolutionDerivative<dim> ESD;
  ESD.set_time(0.0);
  VectorTools::project(dof_handler,
                       constraints,
                       QGauss<dim>(fe.degree + 1),
                       ESD,
                       Solution_v);

  // Creates a folder for data and plots the initial solution
  system("mkdir Solution");
  graph(0);
}

/*---------------------------- Assemble System -------------------------------*/
// Assemble the system at each iteration
template <int dim>
void ElasticWaveEquation<dim>::assemble_nth_iteration()
{
  // Quadrature used
  QGauss<dim> quadrature(fe.degree + 1);

  // Initialises the Mass and Laplace matrices
  MassMatrix.reinit(sparsity_pattern);
  LaplaceMatrix.reinit(sparsity_pattern);
  MatrixCreator::create_mass_matrix(dof_handler,
                                    QGauss<dim>(fe.degree + 1),
                                    MassMatrix);
  MatrixCreator::create_laplace_matrix(dof_handler,
                                    QGauss<dim>(fe.degree + 1),
                                    LaplaceMatrix);

  for (unsigned int i = 1; i < timesteps; i++)
  // for (unsigned int i = 1; i < 2; i++)
  {
  // Initialise: F(t) ~ F_n, F(t+h/2) ~ F_nh, F(t+h)
  F_n.reinit(dof_handler.n_dofs());
  F_nh.reinit(dof_handler.n_dofs());
  F_np.reinit(dof_handler.n_dofs());
  ForcingFunction<dim> forcing_function;
  forcing_function.set_time((i-1)*h);
  VectorTools::create_right_hand_side(dof_handler,
                                      QGauss<dim>(fe.degree + 1),
                                      forcing_function,
                                      F_n);
  forcing_function.set_time((i-0.5)*h);
  VectorTools::create_right_hand_side(dof_handler,
                                      QGauss<dim>(fe.degree + 1),
                                      forcing_function,
                                      F_nh);
  forcing_function.set_time(i*h);
  VectorTools::create_right_hand_side(dof_handler,
                                      QGauss<dim>(fe.degree + 1),
                                      forcing_function,
                                      F_np);

  // Vectors and Matrices needed. This notation comes from the report
  R_u.reinit(dof_handler.n_dofs());
  R_v.reinit(dof_handler.n_dofs());
  R_u_temp1.reinit(dof_handler.n_dofs());
  R_v_temp1.reinit(dof_handler.n_dofs());
  R_u_temp2.reinit(dof_handler.n_dofs());
  R_v_temp2.reinit(dof_handler.n_dofs());
  R_u_temp3.reinit(dof_handler.n_dofs());
  W_u.reinit(dof_handler.n_dofs());
  W_v.reinit(dof_handler.n_dofs());
  W_v_temp.reinit(dof_handler.n_dofs());
  U_temp.reinit(dof_handler.n_dofs());
  V_temp.reinit(dof_handler.n_dofs());

  LHS_Matrix_u.reinit(sparsity_pattern);
  LHS_Matrix_v.reinit(sparsity_pattern);
  LHS_Matrix_u_temp.reinit(sparsity_pattern);
  LHS_Matrix_v_temp.reinit(sparsity_pattern);
  RHS_u.reinit(dof_handler.n_dofs());
  RHS_v.reinit(dof_handler.n_dofs());

  // Constructing the right hand side of known terms
  // R_u
  MassMatrix.vmult(R_u,Solution_u);
  LaplaceMatrix.vmult(R_u_temp1,Solution_u);
  MassMatrix.vmult(R_u_temp2,Solution_v);
  LaplaceMatrix.vmult(R_u_temp3,Solution_v);
  R_u.add(-h*h/2.0,R_u_temp1);
  R_u.add(h,R_u_temp2,-h*h*h/6.0,R_u_temp3);
  R_u.add(h*h/6.0,F_n,h*h/3.0,F_nh);
  // R_v
  MassMatrix.vmult(R_v,Solution_v);
  LaplaceMatrix.vmult(R_v_temp1,Solution_v);
  LaplaceMatrix.vmult(R_v_temp2,Solution_u);
  R_v.add(-h*h/2.0,R_v_temp1,-h,R_v_temp2);
  R_v.add(h/6.0,F_n,2.0*h/3.0,F_nh);
  R_v.add(h/6.0,F_np);

  // W_n & Z_n (missing coefficients in both)
  LaplaceMatrix.vmult(W_u,Solution_u);
  W_u *= h*h*h*h/24.0;
  W_u.add(-h*h*h*h/24.0,F_n);
  LaplaceMatrix.vmult(W_v,Solution_v);
  LaplaceMatrix.vmult(W_v_temp,Solution_u);
  W_v *= h*h*h*h/24.0;
  W_v.add(h*h*h/6.0,W_v_temp);
  W_v.add(-h*h*h/12.0,F_n,-h*h*h/12.0,F_nh);

  // Constructing the left hand side matrix
  LHS_Matrix_u_temp.copy_from(MassMatrix);
  LHS_Matrix_v_temp.copy_from(MassMatrix);

  // Apply boundary conditions U_temp
  Boundary<dim> Boundary_u;
  std::map<types::global_dof_index, double> Boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Boundary_u,
                                           Boundary_values);

  // Apply boundary conditions for V_temp
  BoundaryDerivative<dim> Boundary_v;
  std::map<types::global_dof_index, double> BoundaryDerivative_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Boundary_v,
                                           BoundaryDerivative_values);

  MatrixTools::apply_boundary_values(Boundary_values,
                                     LHS_Matrix_u_temp,
                                     U_temp,
                                     W_u);

  MatrixTools::apply_boundary_values(BoundaryDerivative_values,
                                     LHS_Matrix_v_temp,
                                     V_temp,
                                     W_v);

  // Solve for U_temp and V_temp
  SolverControl solver_stuff_u_temp(1000, 1e-8 * W_u.l2_norm());
  SolverControl solver_stuff_v_temp(1000, 1e-8 * W_v.l2_norm());
  SolverCG<Vector<double>> solver_u_temp(solver_stuff_u_temp);
  SolverCG<Vector<double>> solver_v_temp(solver_stuff_v_temp);

  solver_u_temp.solve(LHS_Matrix_u_temp,
                      U_temp,
                      W_u,
                      PreconditionIdentity());
  solver_v_temp.solve(LHS_Matrix_v_temp,
                      V_temp,
                      W_v,
                      PreconditionIdentity());

  // Construct RHS_u and RHS_v
  LaplaceMatrix.vmult(RHS_u,U_temp);
  LaplaceMatrix.vmult(RHS_v,V_temp);
  RHS_u += R_u;
  RHS_v += R_v;

  // Constructing the left hand side matrix
  LHS_Matrix_u.copy_from(MassMatrix);
  LHS_Matrix_v.copy_from(MassMatrix);

  MatrixTools::apply_boundary_values(Boundary_values,
                                     LHS_Matrix_u,
                                     Solution_u,
                                     RHS_u);

  MatrixTools::apply_boundary_values(BoundaryDerivative_values,
                                     LHS_Matrix_v,
                                     Solution_v,
                                     RHS_v);

  // Solve for Solution
  SolverControl solver_stuff_u(1000, 1e-8 * RHS_u.l2_norm());
  SolverControl solver_stuff_v(1000, 1e-8 * RHS_v.l2_norm());
  SolverCG<Vector<double>> solver_u(solver_stuff_u);
  SolverCG<Vector<double>> solver_v(solver_stuff_v);

  solver_u.solve(LHS_Matrix_u,
                 Solution_u,
                 RHS_u,
                 PreconditionIdentity());
  solver_v.solve(LHS_Matrix_v,
                 Solution_v,
                 RHS_v,
                 PreconditionIdentity());

  graph(i);
  std::cout << "Iteration " << i << std::endl;
  std::cout << "at time " << (i-1)*h << std::endl;
  } // end the forloop
}

/*--------------------------------- Graph ------------------------------------*/
template <int dim>
void ElasticWaveEquation<dim>::graph(const unsigned int time_step)
{
  DataOut<dim> dataout;
  dataout.attach_dof_handler(dof_handler);
  dataout.add_data_vector(Solution_u, "Solution");
  dataout.build_patches();

  const std::string filename = "Solution/solution-" + Utilities::int_to_string(time_step,3) + ".vtu";
  std::ofstream output(filename);
  dataout.write_vtu(output);
}

/*---------------------------------- Run -------------------------------------*/
template <int dim>
void ElasticWaveEquation<dim>::run()
{
  create_grid();
  // create_grid_out(0);
  initialise_system();
  assemble_nth_iteration();
}

/*----------------------------------------------------------------------------*/
/*--------------------------------- Main  ------------------------------------*/
/*----------------------------------------------------------------------------*/
int main()
{
  ElasticWaveEquation<2> Wave;
  Wave.run();
}
