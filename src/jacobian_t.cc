#include "dual.hh"
#include "jacobian.hh"

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"

using jac::Dual;
using jac::dual_array;
using jac::make_jacobian;

// f1 :: dual_array<1> -> Dual<1>
// f1(x) = x^2 + 2x

// f1' :: dual_array<1> -> Dual<1>
// f1'(x) = 2 x + 2
inline Dual<1>
f1(dual_array<1> const& args)
{
  auto const x = args[0];
  return x * (x + 2.0);
}

TEST_CASE("function of one variable")
{
  dual_array<1> x{Dual<1>(1.5)};
  Dual<1> res = f1(x);
  CHECK(res[0] == 5.25);

  // Create the callable derivative object f1deriv:
  jac::Jacobian<decltype(f1), 1> f1deriv(f1);

  // Now evaluate it at the location x.
  Dual<1> gradient = f1deriv(x);
  CHECK(gradient[0] == 5.25);
  CHECK(gradient[1] == 5.0);
}

// f2 is a function template, rather than a function. The idea is that the user
// can instantiate this function with:
//   1)  T = double to yield a function that uses an array of two doubles as its
//   argument type and which returns a double, or
//   2) T = Dual<2> to yield a function that uses an array of two Dual<2> as its
//   argument type and which returns a Dual<2>.
//
// f2 :: array<T, 2> -> T
// f2(x, y) = 2 x - y^3
// df2/dx = 2
// df2/dy = - 3 y^2
template <typename T>
T
f2(std::array<T, 2> x)
{
  return 2 * x[0] - x[1] * x[1] * x[1];
}

TEST_CASE("function of two variables")
{
  using D2 = Dual<2>; // type alias to simplify the following.

  // Create the callable derivative object, f2deriv. Note that we need to
  // specify that we want to instantiate the function template f2 with type D2 =
  // Dual<2>.
  jac::Jacobian<decltype(f2<D2>), 2> f2deriv(f2<D2>);

  // Now evaluate the function and its derivates at the point x.
  dual_array<2> x{D2(2.0), D2(3.0)};
  D2 gradient = f2deriv(x);
  CHECK(gradient[0] == -23.0);
  CHECK(gradient[1] == 2.0);
  CHECK(gradient[2] == -27.0);
}

TEST_CASE("deduction of template parameters")
{
  // This is the way I would *like* make_jacobian to work.
  // The current solution is a cheat that only works for N==2.
  auto fderiv = make_jacobian(f2<Dual<2>>);
}
