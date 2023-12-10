#pragma once

#include <array>
#include <cassert>
#include <initializer_list>
#include <ostream>
#include <stdexcept>

// Struct template Dual is built for automatic computation of the first order
// derivatives of a function. Struct Dual<N> will be used for N-variate
// functions.
//
// We will declare the overload operators +,-,*,/,fabs,<,<=,>,>= etc
// and other functions like sqrt, sin, cos etc outside the class
//
// Forward declaration for the templated class Dual and related templated
// functions

namespace jac {

  // Forward declaration of class template Dual, which represents a dual number
  // with N+1 "slots", suitable for the representation of an N-dimensional
  // derivative. Note that N must be greater than or equal to 1.
  template <typename T, std::size_t N>
  struct Dual;

  // Addition of Duals
  template <typename T, std::size_t N>
  Dual<T, N> operator+(Dual<T, N> const& x, Dual<T, N> const& y);

  // Subtraction of Duals
  template <typename T, std::size_t N>
  Dual<T, N> operator-(Dual<T, N> const& x, Dual<T, N> const& y);

  // Multiplication of Duals
  template <typename T, std::size_t N>
  Dual<T, N> operator*(Dual<T, N> const& x, Dual<T, N> const& y);

  // Division of Duals
  template <typename T, std::size_t N>
  Dual<T, N> operator/(Dual<T, N> const& x, Dual<T, N> const& y);

  // Equality testing

  template <typename T, std::size_t N>
  bool operator==(Dual<T, N> const& a, Dual<T, N> const& b);

  // Write to std::ostream
  template <typename T, std::size_t N>
  std::ostream& operator<<(std::ostream& os, Dual<T, N> const& d);

  template <typename T, std::size_t N>
  struct Dual {
    // for computing the value, Jacobian and an n-variate function
    // f(x1,x2,..xN)

    // Member types.
    using inner_type = T;

    // Static members.

    // values to be used as 'zero' and 'one' with the class.
    // a 'zero' has 'zeros' for *all* members.
    // a 'one' has a 'one' for the first member, and 'zeros' for all others.
    static constexpr inner_type inner_type_zero = {0.0};
    static constexpr inner_type inner_type_one = {1.0};

    // Factory function to create a Dual that is appropriate for the idx'th
    // argument of a multi-argument function, when Duals are being used to
    // calculate derivatives.
    static Dual<T, N> for_deriv(inner_type const& x, std::size_t idx);

    // Factory function to create a Dual from a braced-initializer-list of the
    // next lower-level Dual.
    static Dual<T, N> from_init(std::initializer_list<T> init);

    // Member functions.
    std::size_t constexpr size() const;

    // v[0] is value of the function f(x1,,x2,..xN)
    // v[1..N] are first order derivatives wrt x1,x2,..xN
    //         i.e. df/dx1, df/dx2..df/dxN

    // Access to entries in the Dual.
    inner_type operator[](std::size_t i) const;
    inner_type& operator[](std::size_t i);

    //  Arithmetic assignment operators.
    Dual<T, N>& operator+=(Dual<T, N> const& x);
    Dual<T, N>& operator-=(Dual<T, N> const& x);

    // Data member.
    // Note that all entries of the array are initialized to zero by default, by
    // this declaration of the data member.
    std::array<T, N + 1> v{0.0};
  };

  class exception : public std::runtime_error {
  public:
    explicit exception(char const* msg) : std::runtime_error(msg) {}
  };

  //---------------------------------------------------------------------------
  // Implementation.

  template <typename T, std::size_t N>
  Dual<T, N>
  Dual<T, N>::for_deriv(inner_type const& x, std::size_t idx)
  {
    assert(idx != 0);
    assert(idx <= N);
    Dual<T, N> result{x};
    result[idx] = inner_type_one;
    return result;
  }

  template <typename T, std::size_t N>
  Dual<T, N>
  Dual<T, N>::from_init(std::initializer_list<T> init)
  {
    if (init.size() != N + 1) {
      throw exception("Wrong length initializer list in Dual::from_init");
    }
    Dual<T, N> result;
    std::size_t i = 0;
    for (auto val : init) {
      result[i] = val;
      ++i;
    }
    return result;
  }

  template <typename T, std::size_t N>
  std::size_t constexpr Dual<T, N>::size() const
  {
    return N + 1;
  }

  template <typename T, std::size_t N>
  Dual<T, N>::inner_type
  Dual<T, N>::operator[](std::size_t i) const
  {
    assert(i < N + 1);
    return v[i];
  }

  template <typename T, std::size_t N>
  Dual<T, N>::inner_type&
  Dual<T, N>::operator[](std::size_t i)
  {
    assert(i < N + 1);
    return v[i];
  }

  template <typename T, std::size_t N>
  Dual<T, N>&
  Dual<T, N>::operator+=(Dual<T, N> const& x)
  {
    for (std::size_t i = 0; i != size(); ++i) {
      v[i] += x[i];
    }
    return *this;
  }

  template <typename T, std::size_t N>
  Dual<T, N>&
  Dual<T, N>::operator-=(Dual<T, N> const& x)
  {
    for (std::size_t i = 0; i != size(); ++i) {
      v[i] -= x[i];
    }
    return *this;
  }

  // Free function template implementations.
  template <typename T, std::size_t N>
  Dual<T, N>
  operator+(Dual<T, N> const& x, Dual<T, N> const& y)
  {
    Dual<T, N> result(x);
    result += y;
    return result;
  }

  template <typename T, std::size_t N>
  Dual<T, N>
  operator-(Dual<T, N> const& x, Dual<T, N> const& y)
  {
    Dual<T, N> result(x);
    result -= y;
    return result;
  }

  template <typename T, std::size_t N>
  Dual<T, N>
  operator*(Dual<T, N> const& x, Dual<T, N> const& y)
  {
    Dual<T, N> result;
    result[0] = x[0] * y[0];
    // compute the first derivatives - duv/dxi = udv/dxi +  vdu/dxi
    for (std::size_t i = 1; i != x.size(); ++i) {
      result[i] = x[0] * y[i] + y[0] * x[i];
    }
    return result;
  }

  template <typename T, std::size_t N>
  Dual<T, N>
  operator/(Dual<T, N> const& x, Dual<T, N> const& y)
  {
    T temp = 1.0 / y[0];          // 1/v
    T temp2 = x[0] * temp * temp; // u/sqr(v)
    Dual<T, N> result;

    result.v[0] = x[0] * temp; // u/v
    // compute first derivates d(u/v)/dxi = 1/v*du/dxi - u/sqr(v)*dv/dxi;
    for (std::size_t i = 1; i <= N; i++) {
      result.v[i] = temp * x[i] - temp2 * y[i];
    }

    return result;
  }

  template <typename T, std::size_t N>
  bool
  operator==(Dual<T, N> const& a, Dual<T, N> const& b)
  {
    return a.v == b.v;
  }

  template <typename T, std::size_t N>
  std::ostream&
  operator<<(std::ostream& os, Dual<T, N> const& d)
  {
    os << '(' << d[0];
    for (std::size_t i = 1; i != d.size(); ++i) {
      os << ' ' << d[i];
    }
    os << ')';
    return os;
  }
}