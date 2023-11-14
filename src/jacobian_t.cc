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
  CHECK(deriv[0] == 3.0);
  CHECK(deriv[1] == 4.0);
}

template <typename... Args>
struct DT;

TEST_CASE("type deduction works on expected function")
{
  // Note: this test demonstrates that code is correct just by
  // *compiling*; there are no runtime tests.
  using D2 = Dual<2>;

  // Make sure that f2 has the type we expect. It is a function, not a
  // pointer-to-function nor a reference-to-function.
  using f2_t = decltype(f2<D2>);
  static_assert(std::is_same_v<f2_t, Dual<2>(dual_array<2> const&)>);

  // f2_t is detected correctly by the standard library std::is_function.
  static_assert(std::is_function_v<f2_t>);

  // f2_t objects can be called with argument type dual_array<2> const&, and
  // return Dual<2>. Note that this is different from detecting the signature;
  // this is a test for what call is allowed. Such a call might involve a
  // conversion. The second test tries such a conversion: we can pass an object,
  // rather than a reference, as the argument.
  static_assert(std::is_invocable_r_v<Dual<2>, f2_t, dual_array<2> const&>);
  static_assert(std::is_invocable_r_v<Dual<2>, f2_t, dual_array<2>>);

  // Now that we know that f2_t behaves as expected, let's verify that our
  // compile-time function jac::arg_type works.
  using arg_t = typename jac::arg_type<f2_t&>::type;
  static_assert(std::is_same_v<arg_t, jac::dual_array<2> const&>);
}

TEST_CASE("type deduction works on objects returned by make_jacobian")
{
  // Note: this test demonstrates that code is correct just by
  // *compiling*; there are no runtime tests.
  using D2 = Dual<2>;

  // Make sure that make_jacobian(f2<D2>) has the argument type we expect.
  auto fderiv = make_jacobian(f2<D2>);
  using fderiv_t = decltype(fderiv);

  // fderiv_t is *not* a function type.
  static_assert(not std::is_function_v<fderiv_t>);

  // fderiv_t is callable with the expected argument types, returning the
  // expected type.
  static_assert(std::is_invocable_r_v<Dual<2>, fderiv_t, dual_array<2> const&>);
  static_assert(std::is_invocable_r_v<Dual<2>, fderiv_t, dual_array<2>>);

  // Make sure that our compile-time function jac::arg_type works.
  // using arg_t = typename jac::arg_type<fderiv_t&>::type;
}
