# Inside src/CMakeLists.txt
add_executable(run main.cpp ekf.hpp ekf.cpp)
target_link_libraries(run PRIVATE Eigen3::Eigen)
