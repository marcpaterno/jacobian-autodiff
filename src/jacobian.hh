#pragma once

#include "dual.hh"
#include <array>
#include <concepts>
#include <functional>

namespace jac {

  // dual_array<n> is a type alias to be used as the argument type for a
  // function of n variables.
  template <int N>
  using dual_array = std::array<jac::Dual<N>, N>;

  // jacobian is a class template that impplements the Jacobian of a function
  // through automatic differentiation.
  // If f is an object of type F, where F is callable with function signature:
  //      dual<N> f(dual_array<N> const& x);
  // then jacobian<F> is a type that is callable with function signature:
  //      dual_array<N> g(dual_array<N> const& x);
  // where the 0th slot of the returned value is the value of the function f at
  // x, and the i'th slot (for 1..N) of the returned value are the values of the
  // partial derivatives of f with respect to x_i.

  template <typename F, int N>
  requires std::regular_invocable<F, dual_array<N>> class Jacobian {
  public:
    // construct the jacobian object for the function f.
    explicit Jacobian(F&& f) : func(std::forward<F&&>(f)) {}

    // Evaluate the jacobian of f at point x.
    Dual<N>
    operator()(dual_array<N> const& x) const
    {
      // We make a local copy so that we can modify our copy without altering
      // the original. We should consider having the function take the
      // dual_array by value rather than reference, which will make the copy
      // happen before the call and perhaps allow the compiler to take advantage
      // of the information that no aliasing can be happening.
      dual_array<N> args(x);
      for (std::size_t i = 1; i <= N; ++i) {
        Dual<N>& current_arg = args[i - 1];
        current_arg.v[i] = 1.0;
      }
      return func(args);
    }

  private:
    using f_return_type = Dual<N>;
    using f_arg_type = dual_array<N>;
    std::function<f_return_type(f_arg_type)> func;
  };

  // deduce_n is a compile-time function that calculates the value of N that is
  // associated with the function of type F.
  template <typename F>
  struct deduce_n;

  // The free function template make_jacobian exists to make it easier to create
  // a Jacobian object. We should look at creating appropriate template
  // deduction guides to make this superfluous.
  template <typename F>
  Jacobian<F, deduce_n<F>::value> make_jacobian(F&& func);

  //-------------------------------------------------------------------
  // Implementation below

  // ArgType is a compile-time function to deduce the argument type of the
  // callable type F. The answer is returned as a nested type, 'type'. Note that
  // ArgType can not be used on an overload set, but only on a single function.
  // That function can be the result of the instantiation of a template.

  // General case, which is an incomplete type.
  template <class F>
  struct ArgType;

  // Specialization for a refernce to a callable that takes an argument of type
  // T and returns a value of type R.
  template <class R, class T>
  struct ArgType<R (&)(T)> {
    typedef std::remove_cv_t<std::remove_reference_t<T>> type;
  };

  template <typename F>
  struct deduce_n {
    // Given a callable F, which takes an argument type of:
    //   dual_array<N> const&
    // we want to deduce the value of N.

    using arg_t = typename ArgType<F>::type;
    using value_type = typename arg_t::value_type;
    static int const value = sizeof(arg_t) / sizeof(value_type);
  };

  template <typename F>
  jac::Jacobian<F, deduce_n<F>::value>
  make_jacobian(F&& func)
  {
    return jac::Jacobian<F, deduce_n<F>::value>(func);
  }
}
