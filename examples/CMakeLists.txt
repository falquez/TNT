# Example applications

add_executable( center center.cpp )
add_executable( mps mps.cpp )
add_executable( mps_single mps_single.cpp )

target_link_libraries( center TNT )
target_link_libraries( mps TNT )
target_link_libraries( mps_single TNT )
