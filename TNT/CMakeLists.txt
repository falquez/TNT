add_library(TNT
    src/algebra/sparse/tensorEigen.cpp
    src/algebra/sparse/tensorQRD.cpp
    src/algebra/sparse/tensorSVD.cpp
    src/algebra/tensorEigen.cpp
    src/algebra/tensorMult.cpp
    src/algebra/tensorQRD.cpp
    src/algebra/tensorSVD.cpp
    src/algebra/tensorTranspose.cpp
    src/configuration/configuration.cpp
    src/configuration/observables.cpp
    src/configuration/operator.cpp
    src/configuration/parameters.cpp
    src/extern/blas.cpp
    src/extern/lapack.cpp
    src/extern/primme.cpp
    src/network/mps/mps.cpp
    src/network/mps/sweep.cpp
    src/network/network.cpp    
    src/operator/sparse/mpo.cpp
    src/operator/mpo.cpp
    src/operator/observable.cpp
    src/operator/projection.cpp
    src/parser/parser.cpp
    src/storage/storage.cpp
    src/tensor/sparse/contraction.cpp
    src/tensor/sparse/eigensolver.cpp
    src/tensor/sparse/tensor.cpp
    src/tensor/contraction.cpp
    src/tensor/eigensolver.cpp
    src/tensor/tensor.cpp
    src/tensor/vector.cpp
    src/util/util.cpp
    )

#Include directories for this library.
target_include_directories(
  TNT
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
  PRIVATE src
  )

target_compile_features(TNT PRIVATE cxx_std_17)

target_link_libraries(
  TNT
  PUBLIC tbb hdf5 Boost::system
  PRIVATE openblas primme tcl hptt Boost::filesystem
)

#Unit tests
add_executable(
  testTNT
  test/main.cpp
  test/tensor/tensor.cpp
  test/tensor/svd.cpp

)
target_link_libraries(testTNT TNT)
add_test(NAME test COMMAND testTNT)

##############################################
# Installation instructions

include(GNUInstallDirs)
#set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/JSONUtils)

install(TARGETS TNT
    EXPORT TNT-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

#This is required so that the exported target has the name JSONUtils and not jsonutils
set_target_properties(TNT PROPERTIES EXPORT_NAME TNT)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

