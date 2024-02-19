add_executable(dual_t dual_t.cc)
target_link_libraries(dual_t PRIVATE Catch2::Catch2WithMain)
add_test(dual_t dual_t)

add_executable(function_template_demo function_template_demo.cc)

message("CMAKE_CXX_COMPILER_ID is: ${CMAKE_CXX_COMPILER_ID}")

foreach(exe_name IN ITEMS "dual_t")
  target_compile_options(${exe_name} PRIVATE -Wall -Wextra -Wconversion
                                             -pedantic-errors -Wsign-conversion)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${exe_name} PRIVATE -Werror)
  endif()
  if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set_target_properties(${exe_name} PROPERTIES LINK_OPTIONS
                                                   LINKER:-ld_classic)
    endif()
  endif()
endforeach()
