/* ---------------------------------------------------------------------
 *
 * Title: 1D Wave Equation (Part 1 of Math 762 Project)
 * Author: Sergio Chavez
 * Date: Started 22 February 2022
 *
 * ---------------------------------------------------------------------
 */

// Libraries for constructing the geometry/grids
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
// For output imaging files
#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/data_out.h>
// Finite Element stuff and degrees of freedom handler
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
// Quadrature and function files
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
// Matrix and vector tools
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
// Not sure exactly what these are as of now, but may need them
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/base/utilities.h>
// For c++
#include <iostream>
#include <fstream>
#include <cmath>

using namespace dealii;

template <int dim>
class Wave
{
public:
  Wave();
  void run();
private:
  void setup_system();
  // void assemble_system();
  void solve();
  // void refine_grid();
  void output_results() const;

  Triangulation<dim> triangulation;
  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;

  AffineConstraints<double> constraints;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> mass_matrix;
  SparseMatrix<double> matrix_A;
  SparseMatrix<double> matrix_u;

  Vector<double> solution;
  Vector<double> old_solution;
  Vector<double> system_rhs;

  double       time_step;
  double       time;
  unsigned int timestep_number;
  const double lambda;
};

// Initial values but idk what this is (taken from step-23)
//  but we're choosing 0 for u and its derivative.
template <int dim>
class InitialValues : public Function<dim>
{
public:
  virtual double value(const Point<dim> & /*p*/,
                       const unsigned int component = 0) const override
  {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    return 0;
  }
};

// Right-hand side forcing term. For now, let's leave it at 0 as well.
template <int dim>
class RightHandSide : public Function<dim>
{
public:
  virtual double value(const Point<dim> & /*p*/,
                       const unsigned int component = 0) const override
  {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    return 0;
  }
};

// Finally, the boundary values for u
template <int dim>
class BoundaryValues : public Function<dim>
{
public:
  virtual double value(const Point<dim> & p,
                       const unsigned int component = 0) const override
  {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));

    if ((this->get_time() <= 0.5) && (p[0] < 0))
      return std::sin(this->get_time() * 4 * numbers::PI);
    else
      return 0;
  }
};

// This constructor determines the number degree of the polynomials used
template <int dim>
Wave<dim>::Wave()
  : fe(1)
  , dof_handler(triangulation)
  , time_step(1. / 64)
  , time(time_step)
  , timestep_number(1)
  , lambda(0.5)
{}

// Set up the system
template <int dim>
void Wave<dim>::setup_system()
{
  // const Point<dim> left(-2);
  // const Point<dim> right(2);
  // const bool colorize = true;
  // GridGenerator::hyper_rectangle(triangulation,left,right,colorize);
  GridGenerator::hyper_cube(triangulation, -1, 1);
  triangulation.refine_global(2);

  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl;

  dof_handler.distribute_dofs(fe);

  std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  // Here create the matrices from the problem
  mass_matrix.reinit(sparsity_pattern);
  matrix_A.reinit(sparsity_pattern);

  MatrixCreator::create_mass_matrix(dof_handler,
                                    QGauss<dim>(fe.degree + 1),
                                    mass_matrix);

  // Create the other matrix here

  // Solution set up
  solution.reinit(dof_handler.n_dofs());
  old_solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  constraints.close();
}

template <int dim>
void Wave<dim>::solve()
{
  SolverControl            solver_control(1000, 1e-8 * system_rhs.l2_norm());
  SolverCG<Vector<double>> cg(solver_control);

  cg.solve(matrix_u, solution, system_rhs, PreconditionIdentity());

  std::cout << "   u-equation: " << solver_control.last_step()
            << " CG iterations." << std::endl;
}



template <int dim>
void Wave<dim>::output_results() const
{
  std::ofstream output("Image.gnuplot");
  GridOut grid_out;
  grid_out.write_gnuplot(triangulation,output);
}

// Runs everything
template <int dim>
void Wave<dim>::run()
{
  setup_system();
  output_results();
  std::cout << "This is working so far!" << '\n';
}

int main ()
{
  Wave<1> wave_1d;
  wave_1d.run();

  return 0;
}
