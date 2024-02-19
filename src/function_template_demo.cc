#include <array>
#include <cassert>
// This operator+ supplied the addition function needed for std::array
// to be used as an argument type for the 'twice' function template below.
template <typename T, std::size_t N>
std::array<T, N>
operator+(std::array<T, N> x, std::array<T, N> y)
{
  std::array<T, N> result;
  for (std::size_t i = 0; i != N; ++i) {
    result[i] = x[i] + y[i];
  }
  return result;
}

// If T were a large object (say > 16 bytes), it would likely be best to pass
// the arguments x and y by const reference, rather than by value.

template <typename T>
T
f1(T x, T y)
{
  return x + y;
}

// Apply the function func to the arguments x and y, and return twice the
// result. This requires (approxiamtely) that the type F support being
// called with two arguments of type T, that the return value of
// F when so called have type T, and that type T support addtion. The exact
// requirements on the function would allow some conversions to type T. This
// could be controlled more tightly (or perhaps somewhat more loosely) by
// modifications of the function template.
template <typename F, typename T>
T
twice(F&& f, T x, T y)
{
  auto temp = f(x, y);
  return temp + temp;
}

int
main()
{
  double a = twice(f1<double>, 3.5, 1.5);
  assert(a == 10.0);

  // We introduce a type alias for convenience.
  using array_t = std::array<float, 2>;
  array_t b = twice(f1<array_t>, array_t({1.0, 2.0}), array_t({3.0, 4.0}));
  assert(b[0] == 8.0);
  assert(b[1] == 12.0);
}