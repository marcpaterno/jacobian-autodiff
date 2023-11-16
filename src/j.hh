#pragma once

#include <array>
#include <functional>
#include <type_traits>
#include <utility>

namespace jac {

  template <typename T, int N, int M>
  class J {
  public:
    using f_return_t = std::array<T, M>;
    using f_arg_t = std::array<T, N>;

    J(J const&) = delete;

    template <typename F>
    requires std::is_invocable_r_v<f_return_t, F, f_arg_t> explicit J(F&& f)
      : f_(std::forward<F&&>(f))
    {}

  private:
    std::function<f_return_t(f_arg_t)> f_;
  };

  // TODO: The following out-of-line implementation of the constructor template
  // for the class template compiles for g++12 and g++13, but not with LLVM
  // Clang 17, or Apple Clang 15.

  // template <typename T, int M, int N>
  // template <typename F>
  // requires std::is_invocable_r_v<typename J<T, M, N>::f_return_t,
  //                                F,
  //                                typename J<T, M, N>::f_arg_t>
  // J<T, M, N>::J(F&& f) : f_(std::forward<F&&>(f))
  // {}

}