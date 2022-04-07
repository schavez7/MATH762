/*                            2D Wave equation                                */

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
  void create_grid_out(const unsigned int n);
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
  Vector<double>            F_new;
  Vector<double>            F_current;

  SparseMatrix<double>      LHS_Matrix_u;
  SparseMatrix<double>      LHS_Matrix_v;

  Vector<double>            RHS_u;
  Vector<double>            RHS_v;
  Vector<double>            RHS_u_Temp1;
  Vector<double>            RHS_v_Temp1;
  Vector<double>            RHS_u_Temp2;
  Vector<double>            RHS_v_Temp2;
  Vector<double>            Solution_u_old;
  // Vector<double>            Solution_v_old;

  const double              csquared;
  const double              k;
  const unsigned int        timesteps;
};

/*------------------------------ Constructor ---------------------------------*/
template <int dim>
ElasticWaveEquation<dim>::ElasticWaveEquation()
: dof_handler(triangulation)
, fe(FE_Q<dim>(2))
, n_refinements(7)
, csquared(1.0)
// , k(1.0/(2.0*csquared*std::pow(2.0,n_refinements+1.0)))   // k < h/c = 2^(n_refine + 1)/c
, k(1.0/64.0)
, timesteps(200)
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

/*---------------------------- Output GridOut --------------------------------*/
// Outputs the grid for the nth time-step. Using method of lines (the case
// used in this example thus far) will result in the same grid pattern each n.
// This can use in the case of Rothe's method where the grid is adaptive.
template <int dim>
void ElasticWaveEquation<dim>::create_grid_out(const unsigned int n)
{
  GridOut grid_out;
  const std::string filename = "grid-" + Utilities::int_to_string(n) + ".svg";
  std::ofstream out(filename);
  grid_out.write_svg(triangulation, out);
  std::cout << "The grid for the nth time term is grid-n.svg" << std::endl;
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
  {
  // Initialise: F(n) is F_new and F(n-1) is F_current
  F_current.reinit(dof_handler.n_dofs());
  F_new.reinit(dof_handler.n_dofs());
  ForcingFunction<dim> f;
  f.set_time((i-1)*k);
  VectorTools::create_right_hand_side(dof_handler,
                                      QGauss<dim>(fe.degree + 1),
                                      f,
                                      F_current);//,
                                      // constraints);
  f.set_time(i*k);
  VectorTools::create_right_hand_side(dof_handler,
                                      QGauss<dim>(fe.degree + 1),
                                      f,
                                      F_new);//,
                                      // constraints);
  // Vector used to construct the rhs and Matrix used to construct lhs
  RHS_u.reinit(dof_handler.n_dofs());
  RHS_v.reinit(dof_handler.n_dofs());
  RHS_u_Temp1.reinit(dof_handler.n_dofs());
  RHS_v_Temp1.reinit(dof_handler.n_dofs());
  RHS_u_Temp2.reinit(dof_handler.n_dofs());
  RHS_v_Temp2.reinit(dof_handler.n_dofs());
  LHS_Matrix_u.reinit(sparsity_pattern);
  LHS_Matrix_v.reinit(sparsity_pattern);

  // Constructing the right hand side based on the Crank-Nicholson scheme
  MassMatrix.vmult(RHS_u,Solution_u);
  LaplaceMatrix.vmult(RHS_u_Temp1,Solution_u);
  MassMatrix.vmult(RHS_u_Temp2,Solution_v);
  RHS_u.add(-csquared*k*k/4.0,RHS_u_Temp1,k,RHS_u_Temp2);
  RHS_u.add(k*k/4.0,F_new,k*k/4.0,F_current);

  // Constructing the left hand side matrix
  LHS_Matrix_u.copy_from(MassMatrix);
  LHS_Matrix_u.add(csquared*k*k/4.0,LaplaceMatrix);

  // Apply boundary conditions for u
  Boundary<dim> Boundary_u;
  std::map<types::global_dof_index, double> Boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Boundary_u,
                                           Boundary_values);

  Solution_u_old = Solution_u;
  MatrixTools::apply_boundary_values(Boundary_values,
                                     LHS_Matrix_u,
                                     Solution_u,
                                     RHS_u);

  // Solve for u(n) solution first
  SolverControl solver_control(1000, 1e-8 * RHS_u.l2_norm());
  SolverCG<Vector<double>> solver(solver_control);

  // Solution_u = 0;
  solver.solve(LHS_Matrix_u,
               Solution_u,
               RHS_u,
               PreconditionIdentity());

  graph(i);

  // Now do same for v variable right hand side
  MassMatrix.vmult(RHS_v,Solution_v);
  LaplaceMatrix.vmult(RHS_v_Temp1,Solution_u);
  LaplaceMatrix.vmult(RHS_v_Temp2,Solution_u_old);
  RHS_v.add(-csquared*k/2.0,RHS_v_Temp1,-csquared*k/2.0,RHS_v_Temp2);
  RHS_v.add(k/2.0,F_new,k/2.0,F_current);

  LHS_Matrix_v.copy_from(MassMatrix);

  // Apply boundary conditions for du/dt
  BoundaryDerivative<dim> Boundary_v;
  Boundary_values.clear();
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Boundary_v,
                                           Boundary_values);
  MatrixTools::apply_boundary_values(Boundary_values,
                                     LHS_Matrix_v,
                                     Solution_v,
                                     RHS_v);

  // Now solve for v(n)
  // Solution_v_old = Solution_v;
  // Solution_v = 0;
  solver.solve(LHS_Matrix_v,
               Solution_v,
               RHS_v,
               PreconditionIdentity());
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
