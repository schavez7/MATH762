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
// Finite Element stuff and degrees of freedom handler
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_handler.h>
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
  // void solve();
  // void refine_grid();
  void output_results();
  Triangulation<dim> triangulation;

  FE_Q<dim> fe;
  DoFHandler<dim> dof_handler;
};

// This constructor determines the number degree of the polynomials used
template <int dim>
Wave<dim>::Wave()
  : fe(1)
  , dof_handler(triangulation)
{}

// Set up the system
template <int dim>
void Wave<dim>::setup_system()
{
  const Point<dim> left(-2);
  const Point<dim> right(2);
  const bool colorize = true;
  GridGenerator::hyper_rectangle(triangulation,left,right,colorize);
  triangulation.refine_global(2);
}

template <int dim>
void Wave<dim>::output_results()
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
