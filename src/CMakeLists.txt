# Inside src/CMakeLists.txt
add_executable(run main.cpp ekf.hpp ekf.cpp)

# Link Boost libraries
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(run PRIVATE Eigen3::Eigen)
