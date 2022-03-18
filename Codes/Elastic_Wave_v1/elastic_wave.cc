#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/base/utilities.h>

namespace Step23
{
  using namespace dealii;


  template <int dim>
  class WaveEquation
  {
  public:
    WaveEquation();
    void run();

  private:
    void setup_system();
    void solve_u();
    void solve_v();
    void output_results() const;

    Triangulation<dim> triangulation;
    FE_Q<dim>          fe;
    DoFHandler<dim>    dof_handler;

    AffineConstraints<double> constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> laplace_matrix;
    SparseMatrix<double> matrix_u;
    SparseMatrix<double> matrix_v;

    Vector<double> solution_u, solution_v;
    Vector<double> old_solution_u, old_solution_v;
    Vector<double> system_rhs;

    double       time_step;
    double       time;
    unsigned int timestep_number;
    const double theta;
  };



  /* -------------------------- Initial Values U ---------------------------- */
  template <int dim>
  class InitialValuesU : public Function<dim>
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


  /* --------------------------- Initial Values ----------------------------- */
  template <int dim>
  class InitialValuesV : public Function<dim>
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


  /* -------------------------- Right Hand Side ----------------------------- */
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


  /* ------------------------- Boundary Values U ---------------------------- */
  template <int dim>
  class BoundaryValuesU : public Function<dim>
  {
  public:
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override
    {
      (void)component;
      Assert(component == 0, ExcIndexRange(component, 0, 1));

      /* 1-Dimensional */
      if (dim == 1)
      {
        if ((this->get_time() <= 0.5) && (p[0] < 0))
          return std::sin(this->get_time() * 4 * numbers::PI);
        else
          return 0;
      }
      /* 2-Dimensional */
      if (dim == 2)
      {
        if ((this->get_time() <= 0.5) && (p[0] < 0) && (p[1] < 1. / 3) &&
            (p[1] > -1. / 3))
          return std::sin(this->get_time() * 4 * numbers::PI);
        else
          return 0;
      }
    }
  };


  /* ------------------------- Boundary Values V ---------------------------- */
  template <int dim>
  class BoundaryValuesV : public Function<dim>
  {
  public:
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override
    {
      (void)component;
      Assert(component == 0, ExcIndexRange(component, 0, 1));

      /* 1-Dimensional */
      if (dim == 1)
      {
        if ((this->get_time() <= 0.5) && (p[0] < 0))
          return (std::cos(this->get_time() * 4 * numbers::PI) * 4 * numbers::PI);
        else
          return 0;
      }
      /* 2-Dimensional */
      if (dim == 2)
      {
        if ((this->get_time() <= 0.5) && (p[0] < 0) && (p[1] < 1. / 3) &&
            (p[1] > -1. / 3))
          return (std::cos(this->get_time() * 4 * numbers::PI) * 4 * numbers::PI);
        else
          return 0;
      }
    }
  };



  /* ----------------------- WaveEquation Constructor------------------------ */
  template <int dim>
  WaveEquation<dim>::WaveEquation()
    : fe(1)
    , dof_handler(triangulation)
    , time_step(1. / 64)
    , time(time_step)
    , timestep_number(1)
    , theta(0.5)
  {}


  /* ---------------------------- Setup System ------------------------------ */
  template <int dim>
  void WaveEquation<dim>::setup_system()
  {
    if (dim == 1) {
      const double left  = -1.0;
      const double right =  1.0;
      GridGenerator::hyper_cube(triangulation,
                                left,
                                right,
                                false);
    } else {
      const Point<dim> center;
      const double     radius = 1.0;
      GridGenerator::hyper_ball(triangulation,
                                center,
                                radius,
                                true);
    }
    // GridGenerator::hyper_cube(triangulation, -1, 1);
    triangulation.refine_global(6);

    std::cout << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl;

    dof_handler.distribute_dofs(fe);

    std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl
              << std::endl;

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    // Initialize the 3 matrices we need: the mass matrix, the Laplace matrix,
    // the matrix $M + k^2\theta^2 A$ used when solving for $U^n$ in each time
    // step.
    //
    // When setting up these matrices, note that they all make use of the same
    // sparsity pattern object.
    //
    // We call library functions that build the Laplace and mass matrices.
    // All they need is a DoFHandler object and a quadrature formula object that
    // is to be used for numerical integration.
    mass_matrix.reinit(sparsity_pattern);
    laplace_matrix.reinit(sparsity_pattern);
    matrix_u.reinit(sparsity_pattern);
    matrix_v.reinit(sparsity_pattern);

    MatrixCreator::create_mass_matrix(dof_handler,
                                      QGauss<dim>(fe.degree + 1),
                                      mass_matrix);
    MatrixCreator::create_laplace_matrix(dof_handler,
                                         QGauss<dim>(fe.degree + 1),
                                         laplace_matrix);

    // The rest of the function is spent on setting vector sizes to the
    // correct value. The final line closes the hanging node constraints
    // object. Since we work on a uniformly refined mesh, no constraints exist
    // or have been computed (i.e. there was no need to call
    // DoFTools::make_hanging_node_constraints as in other programs), but we
    // need a constraints object in one place further down below anyway.
    solution_u.reinit(dof_handler.n_dofs());
    solution_v.reinit(dof_handler.n_dofs());
    old_solution_u.reinit(dof_handler.n_dofs());
    old_solution_v.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    constraints.close();
  }


  /* ------------------------------ Solve U --------------------------------- */
  // The next two functions deal with solving the linear systems associated
  // with the equations for $U^n$ and $V^n$.
  //
  // One can make little experiments with preconditioners for the two matrices
  // we have to invert. As it turns out, however, for the matrices at hand
  // here, using Jacobi or SSOR preconditioners reduces the number of
  // iterations necessary to solve the linear system slightly, but due to the
  // cost of applying the preconditioner it is no win in terms of run-time. It
  // is not much of a loss either, but let's keep it simple and just do
  // without:
  template <int dim>
  void WaveEquation<dim>::solve_u()
  {
    SolverControl            solver_control(1000, 1e-8 * system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);

    cg.solve(matrix_u, solution_u, system_rhs, PreconditionIdentity());

    std::cout << "   u-equation: " << solver_control.last_step()
              << " CG iterations." << std::endl;
  }


  /* ------------------------------ Solve V --------------------------------- */
  template <int dim>
  void WaveEquation<dim>::solve_v()
  {
    SolverControl            solver_control(1000, 1e-8 * system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);

    cg.solve(matrix_v, solution_v, system_rhs, PreconditionIdentity());

    std::cout << "   v-equation: " << solver_control.last_step()
              << " CG iterations." << std::endl;
  }



  /* --------------------------- Output Results ----------------------------- */
  template <int dim>
  void WaveEquation<dim>::output_results() const
  {
    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution_u, "U");
    data_out.add_data_vector(solution_v, "V");

    data_out.build_patches();

    const std::string filename =
      "solution-" + Utilities::int_to_string(timestep_number, 3) + ".vtu";
    // Like step-15, since we write output at every time step (and the system
    // we have to solve is relatively easy), we instruct DataOut to use the
    // zlib compression algorithm that is optimized for speed instead of disk
    // usage since otherwise plotting the output becomes a bottleneck:
    DataOutBase::VtkFlags vtk_flags;
    vtk_flags.compression_level =
      DataOutBase::VtkFlags::ZlibCompressionLevel::best_speed;
    data_out.set_flags(vtk_flags);
    std::ofstream output(filename);
    data_out.write_vtu(output);
  }



  /* -------------------------------- Run ----------------------------------- */
  template <int dim>
  void WaveEquation<dim>::run()
  {
    setup_system();

    VectorTools::project(dof_handler,
                         constraints,
                         QGauss<dim>(fe.degree + 1),
                         InitialValuesU<dim>(),
                         old_solution_u);
    VectorTools::project(dof_handler,
                         constraints,
                         QGauss<dim>(fe.degree + 1),
                         InitialValuesV<dim>(),
                         old_solution_v);

    // Time loop over all the time steps until we reach ($T=5$ in this case).
    // In each time step solve for $U^n$, using the equation:
    // $(M^n + k^2\theta^2 A^n)U^n = $
    // $(M^{n,n-1} - k^2\theta(1-\theta) A^{n,n-1})U^{n-1} + kM^{n,n-1}V^{n-1}
    // +$ $k\theta [ k \theta F^n + k(1-\theta) F^{n-1} ]$.

    // Note that we use the same mesh for all time steps
    // What we therefore have to do first is to add up
    // $MU^{n-1} - k^2\theta(1-\theta) AU^{n-1} + kMV^{n-1}$ and the forcing
    // terms, and put the result into the <code>system_rhs</code> vector. (For
    // these additions, we need a TEMPORARY vector that we declare before the
    // loop to avoid repeated memory allocations in each time step.)
    Vector<double> tmp(solution_u.size());
    Vector<double> forcing_terms(solution_u.size());

    for (; time <= 1.5; time += time_step, ++timestep_number)
      {
        std::cout << "Time step " << timestep_number << " at t=" << time
                  << std::endl;

        mass_matrix.vmult(system_rhs, old_solution_u);

        mass_matrix.vmult(tmp, old_solution_v);
        system_rhs.add(time_step, tmp);

        laplace_matrix.vmult(tmp, old_solution_u);
        system_rhs.add(-theta * (1 - theta) * time_step * time_step, tmp);

        RightHandSide<dim> rhs_function;
        rhs_function.set_time(time);
        VectorTools::create_right_hand_side(dof_handler,
                                            QGauss<dim>(fe.degree + 1),
                                            rhs_function,
                                            tmp);
        forcing_terms = tmp;
        forcing_terms *= theta * time_step;

        rhs_function.set_time(time - time_step);
        VectorTools::create_right_hand_side(dof_handler,
                                            QGauss<dim>(fe.degree + 1),
                                            rhs_function,
                                            tmp);

        forcing_terms.add((1 - theta) * time_step, tmp);

        system_rhs.add(theta * time_step, forcing_terms);

        // After so constructing the right hand side vector of the first
        // equation, all we have to do is apply the correct boundary
        // values. As for the right hand side, this is a space-time function
        // evaluated at a particular time, which we interpolate at boundary
        // nodes and then use the result to apply boundary values as we
        // usually do. The result is then handed off to the solve_u()
        // function:
        {
          BoundaryValuesU<dim> boundary_values_u_function;
          boundary_values_u_function.set_time(time);

          std::map<types::global_dof_index, double> boundary_values;
          VectorTools::interpolate_boundary_values(dof_handler,
                                                   0,
                                                   boundary_values_u_function,
                                                   boundary_values);

          // The matrix for solve_u() is the same in every time steps, so one
          // could think that it is enough to do this only once at the
          // beginning of the simulation. However, since we need to apply
          // boundary values to the linear system (which eliminate some matrix
          // rows and columns and give contributions to the right hand side),
          // we have to refill the matrix in every time steps before we
          // actually apply boundary data. The actual content is very simple:
          // it is the sum of the mass matrix and a weighted Laplace matrix:
          matrix_u.copy_from(mass_matrix);
          matrix_u.add(theta * theta * time_step * time_step, laplace_matrix);
          MatrixTools::apply_boundary_values(boundary_values,
                                             matrix_u,
                                             solution_u,
                                             system_rhs);
        }
        solve_u();


        // The second step, i.e. solving for $V^n$, works similarly, except
        // that this time the matrix on the left is the mass matrix (which we
        // copy again in order to be able to apply boundary conditions, and
        // the right hand side is $MV^{n-1} - k\left[ \theta A U^n +
        // (1-\theta) AU^{n-1}\right]$ plus forcing terms. Boundary values
        // are applied in the same way as before, except that now we have to
        // use the BoundaryValuesV class:
        laplace_matrix.vmult(system_rhs, solution_u);
        system_rhs *= -theta * time_step;

        mass_matrix.vmult(tmp, old_solution_v);
        system_rhs += tmp;

        laplace_matrix.vmult(tmp, old_solution_u);
        system_rhs.add(-time_step * (1 - theta), tmp);

        system_rhs += forcing_terms;

        {
          BoundaryValuesV<dim> boundary_values_v_function;
          boundary_values_v_function.set_time(time);

          std::map<types::global_dof_index, double> boundary_values;
          VectorTools::interpolate_boundary_values(dof_handler,
                                                   0,
                                                   boundary_values_v_function,
                                                   boundary_values);
          matrix_v.copy_from(mass_matrix);
          MatrixTools::apply_boundary_values(boundary_values,
                                             matrix_v,
                                             solution_v,
                                             system_rhs);
        }
        solve_v();

        // Finally, after both solution components have been computed, we
        // output the result, compute the energy in the solution, and go on to
        // the next time step after shifting the present solution into the
        // vectors that hold the solution at the previous time step. Note the
        // function SparseMatrix::matrix_norm_square that can compute
        // $\left<V^n,MV^n\right>$ and $\left<U^n,AU^n\right>$ in one step,
        // saving us the expense of a temporary vector and several lines of
        // code:
        output_results();

        std::cout << "   Total energy: "
                  << (mass_matrix.matrix_norm_square(solution_v) +
                      laplace_matrix.matrix_norm_square(solution_u)) /
                       2
                  << std::endl;

        old_solution_u = solution_u;
        old_solution_v = solution_v;
      }
  }
} // namespace Step23


int main()
{
  try
    {
      using namespace Step23;

      WaveEquation<1> wave_equation_solver;
      wave_equation_solver.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
