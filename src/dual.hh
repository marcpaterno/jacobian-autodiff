#pragma once

#include <array>
#include <cassert>
#include <ostream>

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
  template <typename T, int N>
  struct Dual;

  // Addition of Duals
  template <typename T, int N>
  Dual<T, N> operator+(Dual<T, N> const& op1, Dual<T, N> const& op2);

  // Subtraction of Duals
  template <typename T, int N>
  Dual<T, N> operator-(Dual<T, N> const& op1, Dual<T, N> const& op2);

  // Multiplication of Duals
  template <typename T, int N>
  Dual<T, N> operator*(Dual<T, N> const& op1, Dual<T, N> const& op2);

  // Equality testing

  template <typename T, int N>
  bool operator==(Dual<T, N> const& a, Dual<T, N> const& b);

  // Write to std::ostream
  template <typename T, int N>
  std::ostream& operator<<(std::ostream& os, Dual<T, N> const& d);

  template <typename T, int N>
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

    // Factor function to create a Dual that is appropriate for the idx'th
    // argument of a multi-argument function, when Duals are being used to
    // calculate derivatives.
    static Dual<T, N> for_deriv(inner_type const& x, int idx);

    // Member functions.
    int constexpr size() const;

    // v[0] is value of the function f(x1,,x2,..xN)
    // v[1..N] are first order derivatives wrt x1,x2,..xN
    //         i.e. df/dx1, df/dx2..df/dxN

    // Access to entries in the Dual.
    inner_type operator[](int i) const;
    inner_type& operator[](int i);

    //  Arithmetic assignment operators.
    Dual<T, N>& operator+=(Dual<T, N> const& x);
    Dual<T, N>& operator-=(Dual<T, N> const& x);

    // Data member.
    // Note that all entries of the array are initialized to zero by default, by
    // this declaration of the data member.
    std::array<T, N + 1> v{0.0};
  };

  //---------------------------------------------------------------------------
  // Implementation.

  template <typename T, int N>
  Dual<T, N>
  Dual<T, N>::for_deriv(inner_type const& x, int idx)
  {
    assert(idx != 0);
    assert(idx <= N);
    Dual<T, N> result{x};
    result[idx] = inner_type_one;
    return result;
  }

  template <typename T, int N>
  int constexpr Dual<T, N>::size() const
  {
    return N + 1;
  }

  template <typename T, int N>
  Dual<T, N>::inner_type
  Dual<T, N>::operator[](int i) const
  {
    assert(i < N + 1);
    return v[i];
  }

  template <typename T, int N>
  Dual<T, N>::inner_type&
  Dual<T, N>::operator[](int i)
  {
    assert(i < N + 1);
    return v[i];
  }

  template <typename T, int N>
  Dual<T, N>&
  Dual<T, N>::operator+=(Dual<T, N> const& x)
  {
    for (int i = 0; i != size(); ++i) {
      v[i] += x[i];
    }
    return *this;
  }

  template <typename T, int N>
  Dual<T, N>&
  Dual<T, N>::operator-=(Dual<T, N> const& x)
  {
    for (int i = 0; i != size(); ++i) {
      v[i] -= x[i];
    }
    return *this;
  }

  // Free function template implementations.
  template <typename T, int N>
  Dual<T, N>
  operator+(Dual<T, N> const& x, Dual<T, N> const& y)
  {
    Dual<T, N> result(x);
    result += y;
    return result;
  }

  template <typename T, int N>
  Dual<T, N>
  operator-(Dual<T, N> const& x, Dual<T, N> const& y)
  {
    Dual<T, N> result(x);
    result -= y;
    return result;
  }

#if 0
  template <typename T>
  Dual<T>
  operator*(Dual<T> const& op1, Dual<T> const& op2)
  {
    Dual<T> result;

    result.v[0] = op1.v[0] * op2.v[0];
    // compute the first derivatives - duv/dxi = udv/dxi +  vdu/dxi
    for (int i = 1; i <= Dual<T>::N; i++) {
      result.v[i] = op1.v[0] * op2.v[i] + op2.v[0] * op1.v[i];
    }

    return result;
  }
#endif

  template <typename T, int N>
  bool
  operator==(Dual<T, N> const& a, Dual<T, N> const& b)
  {
    return a.v == b.v;
  }

  template <typename T, int N>
  std::ostream&
  operator<<(std::ostream& os, Dual<T, N> const& d)
  {
    os << '(' << d[0];
    for (int i = 1; i != d.size(); ++i) {
      os << ' ' << d[i];
    }
    os << ')';
    return os;
  }
}