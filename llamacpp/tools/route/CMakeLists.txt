set(TARGET llama-route)
add_executable(${TARGET} route.cpp router.cpp sglang_client.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama cpp-httplib ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
