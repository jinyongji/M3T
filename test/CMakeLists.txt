CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

# Small offline regression check; no GTest, camera SDK or model weights needed.
# Its caller supplies a fresh output directory for synthetic images.
add_executable(test_offline_conditioning EXCLUDE_FROM_ALL offline_conditioning_test.cpp)
target_link_libraries(test_offline_conditioning PRIVATE m3t)

# Options
# =============================================================================
if (USE_GTEST)

    # Libraries
    # =========================================================================
    find_package(GTest 1.7.0 REQUIRED)
    include_directories(SYSTEM ${GTEST_INCLUDE_DIRS})

    # Define headers and sources
    # =========================================================================
    set(SOURCES
            common_test.cpp
            camera_test.cpp
            body_test.cpp 
            renderer_geometry_test.cpp
            renderer_test.cpp
            viewer_test.cpp
            model_test.cpp
            color_histograms_test.cpp
            modality_test.cpp 
            detector_test.cpp
            optimizer_test.cpp
            refiner_test.cpp
            tracker_test.cpp)

    set(HEADERS common_test.h)

    # Define target
    # =========================================================================
    add_executable(gtest_run ${SOURCES} ${HEADERS})
    target_link_libraries(gtest_run PUBLIC m3t ${GTEST_BOTH_LIBRARIES})
endif ()
