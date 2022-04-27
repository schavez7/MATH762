/*                            2D Wave equation                                */
/*   Version 7: New Runge-Kutta is complete. Next need the elastic problem.   */


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
    ForcingFunction(const unsigned int n_components = dim, const double time = 0.)
      : Function<dim>(n_components, time) {}
    virtual
    double value(const Point<dim> &p, unsigned int component = 0) const override
    {
      double t = this->get_time();
      if (component == 0)
      {
        return std::sin(p[0]) * std::sin(p[1]) * std::sin(t);
      } else
      {
        return 4.0 * std::sin(2.0*p[0]) * std::sin(2.0*p[1]) * std::sin(2.0*t);
      }
    }
};

/*----------------------------------------------------------------------------*/
/*----------------------------- Exact Function -------------------------------*/
/*----------------------------------------------------------------------------*/
template <int dim>
class ExactSolution : public Function<dim>
{
  public:
    ExactSolution(const unsigned int n_components = dim, const double time = 0.)
      : Function<dim>(n_components, time) {}
    virtual
    double value(const Point<dim> &p, unsigned int component = 0) const override
    {
      double t = this->get_time();
      if (component == 0)
      {
        return std::sin(p[0]) * std::sin(p[1]) * std::sin(t);
      } else {
        return std::sin(2.0*p[0]) * std::sin(2.0*p[1]) * std::sin(2.0*t);
      }
    }
};

/*----------------------------------------------------------------------------*/
/*------------------------ Exact Function Derivative -------------------------*/
/*----------------------------------------------------------------------------*/
template <int dim>
class ExactSolutionDerivative : public Function<dim>
{
  public:
    ExactSolutionDerivative(const unsigned int n_components = dim, const double time = 0.)
      : Function<dim>(n_components, time) {}
    virtual
    double value(const Point<dim> &p, unsigned int component = 0) const override
    {
      double t = this->get_time();
      if (component == 0)
      {
        return std::sin(p[0]) * std::sin(p[1]) * std::cos(t);
      } else {
        return 2.0 * std::sin(2.0*p[0]) * std::sin(2.0*p[1]) * std::cos(2.0*t);
      }
    }
};

/*----------------------------------------------------------------------------*/
/*-------------------------------- Boundary ----------------------------------*/
/*----------------------------------------------------------------------------*/
template <int dim>
class Boundary : public Function<dim>
{
  public:
    Boundary(const unsigned int n_components = dim, const double time = 0.)
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
    BoundaryDerivative(const unsigned int n_components = dim, const double time = 0.)
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
  ElasticWaveEquation(const unsigned int number_time_steps,
                      const unsigned int n_refinements,
                      std::string material_model);
  double run();
private:
  void create_grid();
  void initialise_system();
  void apply_model(Vector<double> & result, const Vector<double> & displacement);
  void apply_boundaries(SparseMatrix<double> &LHS,Vector<double> &Solution,
                        Vector<double> &RHS,double time);
  void assemble_nth_iteration();
  void max_error(const unsigned int i);
  void graph(const unsigned int time_step);

  Triangulation<dim>        triangulation;
  DoFHandler<dim>           dof_handler;
  FESystem<dim>             fe;

  DynamicSparsityPattern    dynamic_sparsity_pattern;
  SparsityPattern           sparsity_pattern;

  AffineConstraints<double> constraints;

  Vector<double>            Solution_u;
  Vector<double>            Solution_v;

  SparseMatrix<double>      MassMatrix;
  SparseMatrix<double>      LaplaceMatrix;
  SparseMatrix<double>      SystemMatrix;

  Vector<double>            F_n;
  Vector<double>            F_nh;
  Vector<double>            F_np;

  SparseMatrix<double>      LHS_Matrix_u;
  SparseMatrix<double>      LHS_Matrix_v;

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

  Vector<double>            Errors_Iteration;

  const unsigned int        n_refinements;
  const unsigned int        number_time_steps;
  const double              cfl;
  const double              csquared;
  const double              h;

  std::string               material_model;
  const double              lambda;
  const double              mu;
};

/*------------------------------ Constructor ---------------------------------*/
template <int dim>
ElasticWaveEquation<dim>::ElasticWaveEquation(const unsigned int number_time_steps,
                                              const unsigned int n_refinements,
                                              std::string material_model)
: dof_handler(triangulation)
, fe(FE_Q<dim>(2),dim)
, n_refinements(n_refinements)
, number_time_steps(number_time_steps)
, cfl(0.90)
, csquared(1.0)
, h(cfl/(csquared*std::pow(2.0,n_refinements)))
, material_model(material_model)
, lambda(1.0)
, mu(1.0)
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
  if (n_refinements == 4)
  {
    system("mkdir Solution");
    graph(0);
  }
}

/*------------------------------ Apply_model ---------------------------------*/
// Determine the right-hand-side which depends on the material model used.
template <int dim>
void ElasticWaveEquation<dim>::apply_model(Vector<double> &result,
                                           const Vector<double> &displacement)
{
  // Quadrature used and incorporated in the fe system
  const QGauss<dim> quadrature(fe.degree+1);
  FEValues<dim> fe_values(fe,
                          quadrature,
                          update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);

  FullMatrix<double>   cell_matrix(fe.n_dofs_per_cell(),fe.n_dofs_per_cell());
  SystemMatrix.reinit(sparsity_pattern);
  std::vector<types::global_dof_index> local_dof_indices(fe.n_dofs_per_cell());

  if (material_model == "wave_laplace_easy")
  {
    LaplaceMatrix.vmult(result,displacement);
  }
  if (material_model == "wave_laplace_hard")
  {
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_matrix = 0;
      fe_values.reinit(cell);
      for (const unsigned int i : fe_values.dof_indices())
      {
        for (const unsigned int j : fe_values.dof_indices())
        {
          for (const unsigned int q_index : fe_values.quadrature_point_indices())
          {
            for (unsigned int d = 0; d < dim; ++d)
            {
              cell_matrix(i,j) +=
                (fe_values.shape_grad_component(i,q_index,d)  *
                fe_values.shape_grad_component(j,q_index,d))  *
                fe_values.JxW(q_index);
             } // end n_components loop
          } // end q_index loop
        } // end j loop
      } // end i loop
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(
        cell_matrix,local_dof_indices,SystemMatrix);
    } // end cell for loop
    SystemMatrix.vmult(result,displacement);
  }
  // if (material_model == "hyperelastic")
  // {
  //   for (const auto &cell : dof_handler.active_cell_iterators())
  //   {
  //     cell_matrix = 0;
  //     fe_values.reinit(cell);
  //     for (const unsigned int i : fe_values.dof_indices())
  //     {
  //
  //     }
  //   }
  // }
}

/*--------------------------- Apply Boundaries -------------------------------*/
template <int dim>
void ElasticWaveEquation<dim>::apply_boundaries(SparseMatrix<double> &LHS,
                                                Vector<double> &Solution,
                                                Vector<double> &RHS,
                                                double time)
{
  // Boundary constraints
  ExactSolution<dim>           Boundary;
  ExactSolutionDerivative<dim> Boundary_Derivative;
  std::map<types::global_dof_index,double> Boundary_Derivative_values;

  Boundary_Derivative.set_time(time*h);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Boundary_Derivative,
                                           Boundary_Derivative_values);
  MatrixTools::apply_boundary_values(Boundary_Derivative_values,
                                     LHS,
                                     Solution,
                                     RHS);

  // Solve system
  SolverControl solver_stuff(1000, 1e-8 * RHS.l2_norm());
  SolverCG<Vector<double>> solver(solver_stuff);
  solver.solve(LHS,
               Solution,
               RHS,
               PreconditionIdentity());
}

/*---------------------------- Assemble System -------------------------------*/
// Assemble the system at each iteration
template <int dim>
void ElasticWaveEquation<dim>::assemble_nth_iteration()
{
  // Initialises the Mass and Laplace matrices
  MassMatrix.reinit(sparsity_pattern);
  LaplaceMatrix.reinit(sparsity_pattern);
  MatrixCreator::create_mass_matrix(dof_handler,
                                    QGauss<dim>(fe.degree + 1),
                                    MassMatrix);
  MatrixCreator::create_laplace_matrix(dof_handler,
                                    QGauss<dim>(fe.degree + 1),
                                    LaplaceMatrix);

  // Max error at each iteration initialise
  Errors_Iteration.reinit(number_time_steps-1);

  for (unsigned int i = 1; i < number_time_steps; i++)
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
  Vector<double> K11;
  Vector<double> K12;
  Vector<double> K12_temp;
  Vector<double> K21;
  Vector<double> K22;
  Vector<double> K22_temp;
  Vector<double> K31;
  Vector<double> K32;
  Vector<double> K32_temp;
  Vector<double> K41;
  Vector<double> K42;
  Vector<double> K42_temp;
  Vector<double> temp;

  K11.reinit(dof_handler.n_dofs());
  K12.reinit(dof_handler.n_dofs());
  K12_temp.reinit(dof_handler.n_dofs());
  K21.reinit(dof_handler.n_dofs());
  K22.reinit(dof_handler.n_dofs());
  K22_temp.reinit(dof_handler.n_dofs());
  K31.reinit(dof_handler.n_dofs());
  K32.reinit(dof_handler.n_dofs());
  K32_temp.reinit(dof_handler.n_dofs());
  K31.reinit(dof_handler.n_dofs());
  K32.reinit(dof_handler.n_dofs());
  K32_temp.reinit(dof_handler.n_dofs());
  K41.reinit(dof_handler.n_dofs());
  K42.reinit(dof_handler.n_dofs());
  K42_temp.reinit(dof_handler.n_dofs());
  temp.reinit(dof_handler.n_dofs());
  SparseMatrix<double> LHS_Matrix;


  /* --- First stage ---*/
  K11 += Solution_v;
  apply_model(K12_temp,Solution_u);
  K12_temp *= -1.0;
  K12_temp += F_n;

  LHS_Matrix.reinit(sparsity_pattern);
  LHS_Matrix.copy_from(MassMatrix);

  apply_boundaries(LHS_Matrix,K12,K12_temp,i-1.0);

  /* --- Second Stage --- */
  K21 += Solution_v;
  K21.add(0.5*h,K12);
  temp.reinit(dof_handler.n_dofs());
  temp += Solution_u;
  temp.add(0.5*h,K11);
  apply_model(K22_temp,temp);
  K22_temp *= -1.0;
  K22_temp += F_nh;

  LHS_Matrix.reinit(sparsity_pattern);
  LHS_Matrix.copy_from(MassMatrix);

  apply_boundaries(LHS_Matrix,K22,K22_temp,i-0.5);

  /* --- Third Stage --- */
  K31 += Solution_v;
  K31.add(0.5*h,K22);
  temp.reinit(dof_handler.n_dofs());
  temp += Solution_u;
  temp.add(0.5*h,K21);
  apply_model(K32_temp,temp);
  K32_temp *= -1.0;
  K32_temp += F_nh;

  LHS_Matrix.reinit(sparsity_pattern);
  LHS_Matrix.copy_from(MassMatrix);

  apply_boundaries(LHS_Matrix,K32,K32_temp,i-0.5);

  /* --- Fourth Stage --- */
  K41 += Solution_v;
  K41.add(h,K32);
  temp.reinit(dof_handler.n_dofs());
  temp += Solution_u;
  temp.add(h,K31);
  apply_model(K42_temp,temp);
  K42_temp *= -1.0;
  K42_temp += F_np;

  LHS_Matrix.reinit(sparsity_pattern);
  LHS_Matrix.copy_from(MassMatrix);

  apply_boundaries(LHS_Matrix,K42,K42_temp,i-0.0);

  /* --- Runge-Kutta Evolution --- */
  temp.reinit(dof_handler.n_dofs());
  temp.add(1.0,K11,2.0,K21);
  temp.add(2.0,K31,1.0,K41);
  temp *= (h/6.0);
  temp += Solution_u;
  Solution_u.reinit(dof_handler.n_dofs());
  Solution_u = temp;

  temp.reinit(dof_handler.n_dofs());
  temp.add(1.0,K12,2.0,K22);
  temp.add(2.0,K32,1.0,K42);
  temp *= (h/6.0);
  temp += Solution_v;
  Solution_v.reinit(dof_handler.n_dofs());
  Solution_v = temp;


  // Need to change this, but if you want to print for different refinements
  // change this
  if (n_refinements == 4)
  {
    graph(i);
  }
  max_error(i);
  } // end the forloop
}

/*--------------------------------- Error ------------------------------------*/
template <int dim>
void ElasticWaveEquation<dim>::max_error(const unsigned int i)
{
  // Compute the pointwise maximum error:
  Vector<double> max_error_per_cell(triangulation.n_active_cells());
    {
      MappingQGeneric<dim> mapping(1);
      ExactSolution<dim>   ES;
      ES.set_time(i*h);
      VectorTools::integrate_difference(mapping,
                                        dof_handler,
                                        Solution_u,
                                        ES,
                                        max_error_per_cell,
                                        QIterated<dim>(QGauss<1>(2), 2),
                                        VectorTools::NormType::Linfty_norm);
      Errors_Iteration(i-1) = *std::max_element(max_error_per_cell.begin(),
                                        max_error_per_cell.end());
    }
}

/*--------------------------------- Graph ------------------------------------*/
template <int dim>
void ElasticWaveEquation<dim>::graph(const unsigned int time_step)
{
  std::vector<std::string> Solution_names ={"Wave1","Wave2"};
  // Solution_names.emplace_back("Wave 2");

  DataOut<dim> dataout;
  dataout.attach_dof_handler(dof_handler);
  dataout.add_data_vector(Solution_u,
                          Solution_names);
  dataout.build_patches();

  const std::string filename = "Solution/solution-" + Utilities::int_to_string(time_step,3) + ".vtu";
  std::ofstream output(filename);
  dataout.write_vtu(output);
}

/*---------------------------------- Run -------------------------------------*/
template <int dim>
double ElasticWaveEquation<dim>::run()
{
  create_grid();
  initialise_system();
  assemble_nth_iteration();
  return Errors_Iteration.linfty_norm();
}

/*----------------------------------------------------------------------------*/
/*--------------------------------- Main  ------------------------------------*/
/*----------------------------------------------------------------------------*/
int main()
{
  unsigned int number_time_steps(2);
  unsigned int total_refinements(4);
  std::string   material_model = "wave_laplace_hard";

  Vector<double> All_Errors(total_refinements);

  for (unsigned int k = 4; k < total_refinements+1; k++)
  {
    ElasticWaveEquation<2> Wave(number_time_steps,k,material_model);
    All_Errors(k-1) = Wave.run();
    std::cout << "Refinements: " << k << std::endl;
    std::cout << "   Timestep " << 0.9/(std::pow(2.0,k)) << " with max error " << All_Errors(k-1) << std::endl;
  }
}
