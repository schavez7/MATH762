/*                     2D Wave equation from scratch                          */

/* These will be preseneted in order of appearance */
// Must create triangulation and a grid
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
// DoF handler and finite elements
#include <deal.II/dofs/dof_handler.h>
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

#include <fstream>
#include <iostream>

using namespace dealii;

// Class template starts here.
template <int dim>
class ElasticWaveEquation
{
public:
  ElasticWaveEquation();
  void run();
private:
  void create_grid();
  void create_grid_out(const unsigned int n);
  void setup_system();

  Triangulation<dim>      triangulation;
  DoFHandler<dim>         dof_handler;
  FE_Q<dim>               fe;
  DynamicSparsityPattern  dynamic_sparsity_pattern;
  SparsityPattern         sparsity_pattern;
  SparseMatrix<double>    system_matrix;
  Vector<double>          solution;
  Vector<double>          system_rhs;
};

// Constructor for ElasticWaveEquation class
template <int dim>
ElasticWaveEquation<dim>::ElasticWaveEquation()
: dof_handler(triangulation)
, fe(1) /* For now let's do degree 1 polynomials */
{}

// This creates the grid. In this case, we'll use a circle for now.
template <int dim>
void ElasticWaveEquation<dim>::create_grid()
{
  GridGenerator::hyper_ball(triangulation);
  triangulation.refine_global(5);
}

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

// Set up the system to be solved by locating where the degrees of freedom are
// in fe. Then creates the matrix using a sparse pattern.
template <int dim>
void ElasticWaveEquation<dim>::setup_system()
{
  // Need this to know the which nodes can very and how many.
  dof_handler.distribute_dofs(fe);
  // Using the above information, a system matrix can be produced of the right size.
  dynamic_sparsity_pattern.reinit(dof_handler.n_dofs(),dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern);
  sparsity_pattern.copy_from(dynamic_sparsity_pattern);

  system_matrix.reinit(sparsity_pattern);
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}

// Runs the program as needed
template <int dim>
void ElasticWaveEquation<dim>::run()
{
  create_grid();
  create_grid_out(0);
  setup_system();
}

int main()
{
  ElasticWaveEquation<2> Wave;
  Wave.run();
}
