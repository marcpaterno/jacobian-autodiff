#include "dual.hh"
#include "jacobian.hh"

#include "catch2/catch_test_macros.hpp"

using jac::Dual;
using jac::dual_array;
using jac::make_jacobian;

inline Dual<1>
f1(dual_array<1> const& args)
{
  auto const x = args[0];
  return x * (x + 2.0);
}

template <typename T>
T
f2(std::array<T, 2> const& x)
{
  return 2 * x[0] - x[1] * x[1] * x[1];
}

// DT is a template used to create a compilation error which contains the
// type(s) of the template parameters.
// Use this in order to determine what the compiler deduces a given type to be.
// To use it, just declare an object to be of type DT<T>:
//       DT<some_type_calcluation> x;
// will generate an error message with the deduced value of
// "some_type_calculation".
template <typename... Args>
struct DT;

TEST_CASE("testing callable traitst")
{
  using D2 = Dual<2>;
  using f2_t = decltype(f2<D2>);
  using traits = jac::callable_traits<f2_t>;
  static_assert(jac::callable_traits<f2_t>::arity == 1);
  static_assert(std::is_same_v<jac::callable_traits<f2_t>::template arg_t<0>,
                               jac::dual_array<2> const&>);
  static_assert(std::is_same_v<jac::callable_traits<f2_t>::result_t, D2>);
}

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
  using arg_t = typename jac::arg_type<f2_t>::type;
  static_assert(std::is_same_v<arg_t, jac::dual_array<2> const&>);

  // Make sure it also works for a reference to the function type f2_t.
  using arg_t2 = typename jac::arg_type<f2_t&>::type;
  static_assert(std::is_same_v<arg_t2, jac::dual_array<2> const&>);
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

  using arg_t = typename jac::arg_type<fderiv_t>::type;
  static_assert(std::is_same_v<arg_t, jac::dual_array<2> const&>);

  static_assert(jac::deduce_n<fderiv_t>::value == 2);
  using t1 =
    jac::arg_type<jac::Jacobian<D2 (&)(dual_array<2> const&), 2>>::type;
  static_assert(std::is_same_v<t1, dual_array<2> const&>);
}
