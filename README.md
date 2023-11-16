# Automatic Calculation of Jacobians and Hessians

This repository contains a C++ demonstration of the calculation of
Jacobians and Hessians through automatic differentiation. The automatic
differentiation is achieved through the use of dual numbers.

## Differentiation through the use of dual numbers

Any analytic real function that can be expressed as a Taylor series can
be evaluated with a dual number $x = a + b \epsilon$, with
$\epsilon^2 = 0$ (and $\epsilon \ne 0$ ). The application of this
function results in the calculation of the function and its derivative:

$$
\begin{aligned}
f(x) & = f(a + b \epsilon) \\
     & = f(a) + b \frac{df}{dx}(a) \\
     & = f(a) + b f'(a)
\end{aligned}
$$

If in this equation we use $b=1$, we get from a single evaluation of $f$
(which must be implemented using dual numbers) that the evaluation of
$f(a,1)$ tells us both $f(a)$ and $f'(a)$.

## Nature of the `make_J` function

We will use the term *callable object* rather than *function* in this
description, except when refering to an actual C++ function. A callable
object may be an actual C++ functions, possibly one created by
instantiating a function template. A callable object may also be an
instance of a class or struct that has a function call operator
(`operator()(T...)`) implemented for some type, or sequence of types,
`T`.

We want to work with callable types `F` such that if `f` is an object of
type `F`, it supports a call like:

    template <typename T, int M, int N>
    array<T, M> f(array<T,N> const& x);

## Nature of the `jacobian` function

This library provides a higher-order function `jacobian` that takes as
an argument a function `f` of $n$ variables and which yields a scalar
value. The function `jacobian` returns a function that takes an argument
$n$ variables and which yields $n+1$ values, the value of the function
and of the first partial derivative with respect to each of the $n$
variables.

The function `f` provided to `jacobian` must be implemented in terms of
dual numbers of $n$ dimensions. Note that a dual number of $n$
dimensions can be represented as an array of $n+1$ values.

The function returned by `jacobian` is also implemented in terms of dual
numbers of $n$ dimensions.

## C++ types in the library

In this library dual numbers of dimension $n$ are represented by the
struct template `Dual<n>`. A mathematical function of $n$ arguments must
be implemented as a callable type taking a single argument of type
`dual_array<n>`. `dual_array<n>` is a type alias for
`std::array<Dual<n>, n>`.
