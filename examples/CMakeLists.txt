# Example applications

add_executable( mps mps.cpp )
add_executable( mps_single mps_single.cpp )
#add_executable( mps_sparse mps_sparse.cpp )
#add_executable( mps_single_sparse mps_single_sparse.cpp )
#add_executable( mps_single_expansion_sparse mps_single_sparse.cpp )
add_executable( svd svd.cpp )

target_link_libraries( mps TNT )
target_link_libraries( mps_single TNT )
#target_link_libraries( mps_sparse TNT )
#target_link_libraries( mps_single_sparse TNT )
#target_link_libraries( mps_single_expansion_sparse TNT )

target_link_libraries( svd TNT )
