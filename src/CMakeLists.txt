add_executable(dual_t dual_t.cc)
target_link_libraries(dual_t PRIVATE Catch2::Catch2WithMain)
add_test(dual_t dual_t)

add_executable(jacobian_t jacobian_t.cc)
target_link_libraries(jacobian_t PRIVATE Catch2::Catch2WithMain)
add_test(jacobian_t jacobian_t)
