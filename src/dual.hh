#pragma once

#include <array>
#include <cmath> // we want math functions in the std::namespace
#include <cstdio>
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
  template <int N>
  struct Dual;

  // Addition of Duals and doubles
  template <int N>
  Dual<N> operator+(Dual<N> const& op1, Dual<N> const& op2);

  template <int N>
  Dual<N> operator+(double op1, Dual<N> const& op2);

  template <int N>
  Dual<N> operator+(Dual<N> const& op1, double op2);

  // Subtraction of Duals and doubles
  template <int N>
  Dual<N> operator-(Dual<N> const& op1, Dual<N> const& op2);

  template <int N>
  Dual<N> operator-(double op1, Dual<N> const& op2);

  template <int N>
  Dual<N> operator-(Dual<N> const& op1, double op2);

  template <int N>
  Dual<N> operator-(Dual<N> const& op1);

  // Multiplication of Duals and doubles
  template <int N>
  Dual<N> operator*(Dual<N> const& op1, Dual<N> const& op2);
  template <int N>

  Dual<N> operator*(Dual<N> const& op1, double op2);

  template <int N>
  Dual<N> operator*(double op1, Dual<N> const& op2);

  // Division of Duals and doubles
  template <int N>
  Dual<N> operator/(Dual<N> const& op1, Dual<N> const& op2);

  template <int N>
  Dual<N> operator/(Dual<N> const& op1, double op2);

  template <int N>
  Dual<N> operator/(const double op1, Dual<N> const& op2);

  // Ordering operators for Duals and doubles.
  // Consider replacing all with operator<=> and relying upon implicit
  // conversion of doubles to Dual<N>.
  //
  // equality
  template <int N>
  bool operator==(Dual<N> const& op1, Dual<N> const& op2);

  template <int N>
  bool operator==(Dual<N> const& op1, double op2);

  template <int N>
  bool operator==(double op1, Dual<N> const& op2);

  // less-than ordering
  template <int N>
  bool operator<(Dual<N> const& op1, Dual<N> const& op2);

  template <int N>
  bool operator<(Dual<N> const& op1, double op2);

  template <int N>
  bool operator<(double op1, Dual<N> const& op2);

  // less-equal ordering
  template <int N>
  bool operator<=(Dual<N> const& op1, Dual<N> const& op2);

  template <int N>
  bool operator<=(Dual<N> const& op1, double op2);

  template <int N>
  bool operator<=(const double op1, Dual<N> const& op2);

  // greater-than orderin
  template <int N>
  bool operator>(Dual<N> const& op1, Dual<N> const& op2);

  template <int N>
  bool operator>(Dual<N> const& op1, double op2);

  template <int N>
  bool operator>(double op1, Dual<N> const& op2);

  // greater-equal ordering
  template <int N>
  bool operator>=(Dual<N> const& op1, Dual<N> const& op2);

  template <int N>
  bool operator>=(Dual<N> const& op1, double op2);

  template <int N>
  bool operator>=(const double op1, Dual<N> const& op2);

  // non-algebraic basic functions
  template <int N>
  Dual<N> abs(Dual<N> const& op1);

  template <int N>
  Dual<N> sqrt(Dual<N> const& op1);
  template <int N>

  Dual<N> sin(Dual<N> const& op1);

  template <int N>
  Dual<N> cos(Dual<N> const& op1);

  // stream insertion
  template <int N>
  std::ostream& operator<<(std::ostream& os, Dual<N> const& d);

  // C-style printing support.
  template <int N>
  void printDual(Dual<N> const& f);

  //--------------------------------------------------------------------------------------------
  // Implementations below.

  template <int N>
  struct Dual {
    static_assert(N >= 1, "Dual is defined only for positive integers");
    // for computing the value, Jacobian and an n-variate function
    // f(x1,x2,..xN)

    // Use std::array so that copying, etc. work.
    // Provide default value so that default initialization of Dual gives
    // defined values.
    std::array<double, N + 1> v{0.0};

    // v[0] is value of the function f(x1,,x2,..xN)
    // v[1..N] are first order derivatives wrt x1,x2,..xN
    //         i.e. df/dx1, df/dx2..df/dxN

    // Construct a Dual from an std::array<double, N+1>.
    explicit Dual(std::array<double, N + 1> const& vals) : v(vals) {}

    // Construct a Dual with slot 0 = value, slot i = 1, and all other entries
    // zero. This also provides automatic conversion of a double to a Dual with
    // v[0] set to the value of the double, and all other v[i] set to zero.
    Dual(double value = 0.0, int i = 0)
    {
      // We set v[i] first so that if i==0, we end up with v[0] == value rather
      // than 1.0
      v[i] = 1.0;
      v[0] = value;
    }

    double
    operator[](int i) const
    {
      return v[i];
    }

    double&
    operator[](int i)
    {
      return v[i];
    }
  };

  // Implementation of arithemetic operations.
  template <int N>
  Dual<N>
  operator+(Dual<N> const& op1, Dual<N> const& op2)
  {
    Dual<N> result;
    for (int i = 0; i <= N; i++) {
      result.v[i] = op1.v[i] + op2.v[i];
    }
    return result;
  }

  template <int N>
  Dual<N>
  operator+(double op1, Dual<N> const& op2)
  {
    Dual<N> result;
    result.v[0] = op1 + op2.v[0];
    for (int i = 1; i <= N; i++) {
      result.v[i] = op2.v[i];
    }
    return result;
  }

  template <int N>
  Dual<N>
  operator+(Dual<N> const& op1, double op2)
  {
    Dual<N> result;
    result.v[0] = op1.v[0] + op2;
    for (int i = 1; i <= N; i++) {
      result.v[i] = op1.v[i];
    }
    return (result);
  }

  template <int N>
  Dual<N>
  operator-(Dual<N> const& op1, Dual<N> const& op2)
  {
    Dual<N> result;
    for (int i = 0; i <= N; i++) {
      result.v[i] = op1.v[i] - op2.v[i];
    }
    return (result);
  }

  template <int N>
  Dual<N>
  operator-(double op1, Dual<N> const& op2)
  {
    Dual<N> result;
    result.v[0] = op1 - op2.v[0];
    for (int i = 1; i <= N; i++) {
      result.v[i] = -op2.v[i];
    }
    return (result);
  }

  template <int N>
  Dual<N>
  operator-(Dual<N> const& op1, double op2)
  {
    Dual<N> result;
    result.v[0] = op1.v[0] - op2;
    for (int i = 1; i <= N; i++) {
      result.v[i] = op1.v[i];
    }
    return (result);
  }

  template <int N>
  Dual<N>
  operator-(Dual<N> const& op1)
  {
    Dual<N> result;
    for (int i = 0; i <= N; i++) {
      result.v[i] = -op1.v[i];
    }
    return (result);
    ;
  }

  template <int N>
  Dual<N>
  operator*(Dual<N> const& op1, Dual<N> const& op2)
  {
    Dual<N> result;

    result.v[0] = op1.v[0] * op2.v[0];
    // compute the first derivatives - duv/dxi = udv/dxi +  vdu/dxi
    for (int i = 1; i <= N; i++) {
      result.v[i] = op1.v[0] * op2.v[i] + op2.v[0] * op1.v[i];
    }

    return result;
  }

  template <int N>
  Dual<N>
  operator*(Dual<N> const& op1, double op2)
  {
    Dual<N> result;
    for (int i = 0; i <= N; i++) {
      result.v[i] = op2 * op1.v[i];
    }
    return result;
  }

  template <int N>
  Dual<N>
  operator*(double op1, Dual<N> const& op2)
  {
    Dual<N> result;
    for (int i = 0; i <= N; i++) {
      result.v[i] = op1 * op2.v[i];
    }
    return result;
  }

  template <int N>
  Dual<N>
  operator/(Dual<N> const& op1, Dual<N> const& op2)
  {
    double temp = 1.0 / op2.v[0];          // 1/v
    double temp2 = op1.v[0] * temp * temp; // u/sqr(v)
    Dual<N> result;

    result.v[0] = op1.v[0] * temp; // u/v
    // compute first derivates d(u/v)/dxi = 1/v*du/dxi - u/sqr(v)*dv/dxi;
    for (int i = 1; i <= N; i++) {
      result.v[i] = temp * op1.v[i] - temp2 * op2.v[i];
    }

    return result;
  }

  template <int N>
  Dual<N>
  operator/(Dual<N> const& op1, double op2)
  {
    double temp = 1.0 / op2;
    Dual<N> result;
    for (int i = 0; i <= N; i++) {
      result.v[i] = temp * op1.v[i];
    }
    return result;
  }

  template <int N>
  Dual<N>
  operator/(double op1, Dual<N> const& op2)
  {
    double temp = 1.0 / op2.v[0];
    double temp2 = -op1 * temp * temp;

    Dual<N> result;

    result.v[0] = op1 * temp;
    // compute the first derivatives - d(A/u)/dxi = A*((-1/(u*u)) *du/dxi)
    for (int i = 1; i <= N; i++) {
      result.v[i] = temp2 * op2.v[i];
    }

    return result;
  }

  template <int N>
  bool
  operator==(Dual<N> const& op1, Dual<N> const& op2)
  {
    return op1.v[0] == op2.v[0];
  }

  template <int N>
  bool
  operator==(Dual<N> const& op1, double op2)
  {
    return op1.v[0] == op2;
  }

  template <int N>
  bool
  operator==(double op1, Dual<N> const& op2)
  {
    return op1 == op2.v[0];
  }

  template <int N>
  bool
  operator<(Dual<N> const& op1, Dual<N> const& op2)
  {
    return (op1.v[0] < op2.v[0]);
  }

  template <int N>
  bool
  operator<(Dual<N> const& op1, double op2)
  {
    return (op1.v[0] < op2);
  }

  template <int N>
  bool
  operator<(double op1, Dual<N> const& op2)
  {
    return (op1 < op2.v[0]);
  }

  template <int N>
  bool
  operator<=(Dual<N> const& op1, Dual<N> const& op2)
  {
    return (op1.v[0] <= op2.v[0]);
  }

  template <int N>
  bool
  operator<=(Dual<N> const& op1, double op2)
  {
    return (op1.v[0] <= op2);
  }

  template <int N>
  bool
  operator<=(double op1, Dual<N> const& op2)
  {
    return (op1 <= op2.v[0]);
  }

  template <int N>
  bool
  operator>(Dual<N> const& op1, Dual<N> const& op2)
  {
    return (op1.v[0] > op2.v[0]);
  }

  template <int N>
  bool
  operator>(Dual<N> const& op1, double op2)
  {
    return (op1.v[0] > op2);
  }

  template <int N>
  bool
  operator>(double op1, Dual<N> const& op2)
  {
    return (op1 > op2.v[0]);
  }

  template <int N>
  bool
  operator>=(Dual<N> const& op1, Dual<N> const& op2)
  {
    return (op1.v[0] >= op2.v[0]);
  }

  template <int N>
  bool
  operator>=(Dual<N> const& op1, double op2)
  {
    return (op1.v[0] >= op2);
  }

  template <int N>
  bool
  operator>=(double op1, Dual<N> const& op2)
  {
    return (op1 >= op2.v[0]);
  }

  template <int N>
  Dual<N>
  fabs(Dual<N> const& op1)
  {
    Dual<N> result;
    result.v[0] = fabs(op1.v[0]);
    double sign = (op1.v[0] < 0.0 ? -1.0 : (op1.v > 0.0 ? 1.0 : 0.0));
    for (int i = 1; i <= N; i++) {
      result.v[i] = sign * op1.v[i];
    }
    return result;
  }

  template <int N>
  Dual<N>
  sqrt(Dual<N> const& op1)
  {
    double temp1 = sqrt(op1.v[0]);         // sqrt(u)
    double invt1 = 1.0 / temp1;            // 1/sqrt(u)
    double temp2 = 2.0 * op1.v[0] * temp1; // 2*u*sqrt(u)
    double invt2 = 1.0 / temp2;            // 1/(2*u*sqrt(u))

    Dual<N> result;

    result.v[0] = temp1;

    if (temp1 > 1.0E-15) // this check avoids "division by 0" type errors
    {
      // compute the first derivatives - dsqrt(u)/dxi = 1/2*sqrt(u)du/dxi
      for (int i = 1; i <= N; i++) {
        result.v[i] = 0.5 * invt1 * op1.v[i];
      }
    } else // this is to avoid "division by zero" and/or mavhine precision type
           // errors
      for (int i = 1; i <= N; i++) {
        result.v[i] = 1.0E+15 * op1.v[i];
      }
    return result;
  }

  template <int N>
  Dual<N>
  sin(Dual<N> const& op1)
  {
    Dual<N> result;
    double temp1 = std::sin(op1.v[0]);
    double temp2 =
      std::cos(op1.v[0]); // Can use sqrt(1-temp1*temp1) if it is faster

    result.v[0] = temp1;
    // Compute first derivatives
    for (int i = 1; i <= N; i++) {
      result.v[i] = temp2 * op1.v[i];
    }
    return result;
  }

  template <int N>
  Dual<N>
  cos(Dual<N> const& op1)
  {
    Dual<N> result;
    double temp1 = std::cos(op1.v[0]);
    double temp2 =
      -std::sin(op1.v[0]); // Can use -sqrt(1-temp1*temp1) if it is faster

    result.v[0] = temp1;
    // compute first derivatives
    for (int i = 1; i <= N; i++) {
      result.v[i] = temp2 * op1.v[i];
    }

    return result;
  }

  template <int N>
  std::ostream&
  operator<<(std::ostream& os, Dual<N> const& d)
  {
    // We print out the first value, and then all others are preceeded by a
    // space. This is safe because we always have N >= 1.
    os << d[0];
    for (std::size_t i = 1; i <= N; ++i) {
      os << ' ' << d[i];
    }
    return os;
  }

  // C-style printing support.
  template <int N>
  void
  printDual(Dual<N> const& f)
  {
    std::printf("function value: %f \n", f.v[0]);
    std::printf("Jacobian : ");
    for (int i = 1; i <= N; i++) {
      std::printf("%f ", f.v[i]);
    }
    std::printf("\n");
  }
} // namespace jac

namespace std {
  // Support for std::size and std::ssize are permitted for user-defined types.
  // The size of a Dual<N> is N+1; allowable indices go from 0 to N, inclusive.

  template <int N>
  constexpr std::ptrdiff_t
  ssize(jac::Dual<N> const&)
  {
    return N + 1;
  }

  template <int N>
  constexpr std::size_t
  size(jac::Dual<N> const&)
  {
    return N + 1;
  }
} // namespace std
