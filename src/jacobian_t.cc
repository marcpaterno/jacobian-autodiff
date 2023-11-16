#include "dual.hh"
#include "jacobian.hh"

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"

// DT is a template used to create a compilation error which contains the
// type(s) of the template parameters.
// Use this in order to determine what the compiler deduces a given type to be.
// To use it, just declare an object to be of type DT<T>:
//       DT<some_type_calcluation> x;
// will generate an error message with the deduced value of
// "some_type_calculation".
template <typename... Args>
struct DT;

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
// d(df2/dx)/dx = 0.
// d(df2/dy)/dx = 0.
// d(df2/dx)/dy = 0.
// d(df2/dy)/dy = -6 y
template <typename T>
T
f2(std::array<T, 2> const& x)
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
  static_assert(std::size(gradient) == 3);
  CHECK(gradient[0] == -23.0);
  CHECK(gradient[1] == 2.0);
  CHECK(gradient[2] == -27.0);
}

TEST_CASE("deduction of template parameters")
{
  // Test of deduction using our function template that models a function of two
  // arguments; note that we must explicitly instantiate the template by
  // supplying the template parameter as Dual<2>.
  using D2 = Dual<2>; // type alias to simplify the following.
  auto fderiv = make_jacobian(f2<D2>);

  dual_array<2> x{D2(2.0), D2(3.0)};
  auto gradient = fderiv(x);
  CHECK(gradient[0] == -23.0);
  CHECK(gradient[1] == 2.0);
  CHECK(gradient[2] == -27.0);

  // Test of the deduction with a function that is *not* a template
  // instantiation.
  auto f1_deriv = make_jacobian(f1);
  dual_array<1> x2({Dual<1>(1.0)});
  auto deriv = f1_deriv(x2);
  static_assert(std::size(deriv) == 2);
  CHECK(deriv[0] == 3.0);
  CHECK(deriv[1] == 4.0);
}

TEST_CASE("two applications of jac::make_jacobian")
{
  using D2 = Dual<2>;
  auto fderiv = make_jacobian(f2<D2>);
  // TODO: Fix make_jacobian so that it correctly applies to the return value of
  // make_jacobian.
  auto fhess = make_jacobian(make_jacobian(f2<D2>));
  dual_array<2> x{D2(2.0), D2(3.0)};
  D2 hess = fhess(x);
  // TODO: Fix make_jacobian so that we can use it on an lvalue, not just on an
  // rvalue.
#if 0
  auto fh2 = make_jacobian(fderiv);
#endif
}