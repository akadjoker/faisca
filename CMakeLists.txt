cmake_minimum_required(VERSION 3.16)

project(faisca VERSION 0.2.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(FAISCA_BUILD_CLI "Build the faisca command line tool" ON)
option(FAISCA_BUILD_TESTS "Build the tests" ON)
option(FAISCA_BUILD_BENCH "Build the benchmarks" OFF)
option(FAISCA_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(FAISCA_NATIVE "Compile for the CPU of this machine (-march=native)" ON)
option(FAISCA_VULKAN "Build the Vulkan backend when vulkan/vulkan.h is available" ON)
set(FAISCA_CT_DIR "" CACHE PATH
    "Checkout of github.com/akadjoker/containers, adds the ct candidates to the benchmarks")
set(FAISCA_LLAMA_DIR "" CACHE PATH
    "Checkout of github.com/ggml-org/llama.cpp, adds the comparison with its GGUF loader")
set(FAISCA_WHISPER_DIR "" CACHE PATH
    "Source tree of whisper.cpp (the patched reference copy) for the whisper_layer_dump tool")
set(FAISCA_WHISPER_BUILD "" CACHE PATH
    "Build tree of that whisper.cpp whose bin/ holds libwhisper.so and libggml*.so")

if(MSVC)
    set(FAISCA_WARNINGS /W4 /WX)
else()
    set(FAISCA_WARNINGS -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror)
endif()

if(FAISCA_NATIVE AND NOT MSVC)
    add_compile_options(-march=native)
endif()

# the same float32 results at every optimization level: FMA only where the code asks for it
if(NOT MSVC)
    add_compile_options(-ffp-contract=off)
endif()

if(FAISCA_SANITIZE AND NOT MSVC)
    add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

add_library(faisca_lib STATIC
    src/error.cpp
    src/gguf.cpp
    src/mapped_file.cpp
    src/memory.cpp
    src/model.cpp
    src/ops.cpp
    src/runner.cpp
    src/session.cpp
    src/speculative.cpp
    src/thread.cpp
    src/tokenizer.cpp
    src/unicode.cpp
    src/audio.cpp
    src/ops_whisper.cpp
    src/ops_conv.cpp
    src/whisper_model.cpp
    src/whisper_runner.cpp
    src/pocket_text.cpp
    src/pocket_model.cpp
    src/pocket_runner.cpp
)
find_package(Threads REQUIRED)
target_link_libraries(faisca_lib PUBLIC Threads::Threads)

set(FAISCA_HAVE_VULKAN FALSE)
if(FAISCA_VULKAN)
    include(CheckIncludeFileCXX)
    check_include_file_cxx(vulkan/vulkan.h FAISCA_VULKAN_HEADER)
    if(FAISCA_VULKAN_HEADER)
        set(FAISCA_HAVE_VULKAN TRUE)
        target_sources(faisca_lib PRIVATE src/vulkan/device.cpp src/vulkan/ops.cpp src/vulkan/runner.cpp src/vulkan/shaders_spv.cpp)
        target_compile_definitions(faisca_lib PUBLIC FAISCA_VULKAN=1)
        if(UNIX)
            target_link_libraries(faisca_lib PUBLIC ${CMAKE_DL_LIBS})
        endif()
    else()
        message(STATUS "faisca: vulkan/vulkan.h not found, the Vulkan backend is disabled")
    endif()
endif()
set_target_properties(faisca_lib PROPERTIES OUTPUT_NAME faisca)
target_include_directories(faisca_lib PUBLIC include)
target_compile_options(faisca_lib PRIVATE ${FAISCA_WARNINGS})

if(FAISCA_BUILD_CLI)
    add_executable(faisca_cli cli/main.cpp)
    set_target_properties(faisca_cli PROPERTIES OUTPUT_NAME faisca)
    target_link_libraries(faisca_cli PRIVATE faisca_lib)
    target_compile_options(faisca_cli PRIVATE ${FAISCA_WARNINGS})
endif()

if(FAISCA_BUILD_TESTS)
    enable_testing()

    add_executable(containers_unit tests/containers_unit.cpp)
    target_link_libraries(containers_unit PRIVATE faisca_lib)
    target_compile_options(containers_unit PRIVATE ${FAISCA_WARNINGS})
    add_test(NAME containers_unit COMMAND containers_unit)

    add_executable(gguf_unit tests/gguf_unit.cpp)
    target_link_libraries(gguf_unit PRIVATE faisca_lib)
    target_compile_options(gguf_unit PRIVATE ${FAISCA_WARNINGS})
    add_test(NAME gguf_unit COMMAND gguf_unit)

    add_executable(tokenizer_unit tests/tokenizer_unit.cpp)
    target_link_libraries(tokenizer_unit PRIVATE faisca_lib)
    target_compile_options(tokenizer_unit PRIVATE ${FAISCA_WARNINGS})
    add_test(NAME tokenizer_unit COMMAND tokenizer_unit)

    add_executable(ops_unit tests/ops_unit.cpp)
    target_link_libraries(ops_unit PRIVATE faisca_lib)
    target_compile_options(ops_unit PRIVATE ${FAISCA_WARNINGS})
    add_test(NAME ops_unit COMMAND ops_unit)

    add_executable(runner_dump tests/runner_dump.cpp)
    add_executable(whisper_dump tests/whisper_dump.cpp)
    target_link_libraries(whisper_dump PRIVATE faisca_lib)
    add_executable(pocket_dump tests/pocket_dump.cpp)
    target_link_libraries(pocket_dump PRIVATE faisca_lib)
    target_link_libraries(runner_dump PRIVATE faisca_lib)
    target_compile_options(runner_dump PRIVATE ${FAISCA_WARNINGS})

    add_executable(matmul_check tests/matmul_check.cpp)
    target_link_libraries(matmul_check PRIVATE faisca_lib)

    add_executable(bench_runner bench/bench_runner.cpp)
    target_link_libraries(bench_runner PRIVATE faisca_lib)

    add_executable(speculative_check tests/speculative_check.cpp)
    target_link_libraries(speculative_check PRIVATE faisca_lib)

    add_executable(session_check tests/session_check.cpp)
    target_link_libraries(session_check PRIVATE faisca_lib)
    target_compile_options(session_check PRIVATE ${FAISCA_WARNINGS})
    target_compile_options(matmul_check PRIVATE ${FAISCA_WARNINGS})

    if(FAISCA_HAVE_VULKAN)
        add_executable(vulkan_check tests/vulkan_check.cpp)
        target_link_libraries(vulkan_check PRIVATE faisca_lib)
        target_compile_options(vulkan_check PRIVATE ${FAISCA_WARNINGS})
    endif()

    add_executable(gguf_dump tests/gguf_dump.cpp)
    target_link_libraries(gguf_dump PRIVATE faisca_lib)
    target_compile_options(gguf_dump PRIVATE ${FAISCA_WARNINGS})

    add_executable(tokenizer_dump tests/tokenizer_dump.cpp)
    target_link_libraries(tokenizer_dump PRIVATE faisca_lib)
    target_compile_options(tokenizer_dump PRIVATE ${FAISCA_WARNINGS})

    add_executable(gguf_malformed tests/gguf_malformed.cpp)
    target_link_libraries(gguf_malformed PRIVATE faisca_lib)
    target_compile_options(gguf_malformed PRIVATE ${FAISCA_WARNINGS})

    find_package(Python3 COMPONENTS Interpreter)
    set(FAISCA_HAVE_GGUF_PY FALSE)
    if(Python3_FOUND)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import gguf, numpy"
            RESULT_VARIABLE FAISCA_GGUF_PY_RESULT
            OUTPUT_QUIET ERROR_QUIET
        )
        if(FAISCA_GGUF_PY_RESULT EQUAL 0)
            set(FAISCA_HAVE_GGUF_PY TRUE)
        endif()
    endif()

    if(FAISCA_HAVE_GGUF_PY)
        set(FAISCA_TEST_FILES_DIR ${CMAKE_BINARY_DIR}/test_files)
        set(FAISCA_TEST_FILES
            ${FAISCA_TEST_FILES_DIR}/basic.gguf
            ${FAISCA_TEST_FILES_DIR}/all_tensor_types.gguf
            ${FAISCA_TEST_FILES_DIR}/alignment64.gguf
            ${FAISCA_TEST_FILES_DIR}/no_tensors.gguf
            ${FAISCA_TEST_FILES_DIR}/llama_like.gguf
        )

        add_test(NAME gguf_reference_files
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/make_test_gguf.py ${FAISCA_TEST_FILES_DIR})
        set_tests_properties(gguf_reference_files PROPERTIES FIXTURES_SETUP reference_files)

        add_test(NAME gguf_compare_with_gguf_py
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/compare_with_gguf_py.py
                    $<TARGET_FILE:gguf_dump> ${FAISCA_TEST_FILES})
        set_tests_properties(gguf_compare_with_gguf_py PROPERTIES FIXTURES_REQUIRED reference_files)

        add_test(NAME gguf_malformed COMMAND gguf_malformed ${FAISCA_TEST_FILES})
        set_tests_properties(gguf_malformed PROPERTIES FIXTURES_REQUIRED reference_files)
    else()
        message(STATUS "faisca: python3 with gguf and numpy not found, reference tests are disabled "
                       "(pip install gguf numpy)")
    endif()
endif()

if(FAISCA_BUILD_BENCH)
    set(FAISCA_LOADER_BENCH_SOURCES
        bench/bench_loader.cpp
        src/error.cpp
        src/gguf.cpp
        src/mapped_file.cpp
    )

    add_executable(bench_loader_faisca ${FAISCA_LOADER_BENCH_SOURCES})
    target_include_directories(bench_loader_faisca PRIVATE include)
    target_compile_definitions(bench_loader_faisca PRIVATE "FAISCA_BENCH_NAME=\"faisca containers\"")

    add_executable(bench_loader_std ${FAISCA_LOADER_BENCH_SOURCES})
    target_include_directories(bench_loader_std PRIVATE include bench/adapters)
    target_compile_definitions(bench_loader_std PRIVATE
        "FAISCA_BENCH_NAME=\"std containers\""
        "FAISCA_BENCH_CONTAINERS_HEADER=\"std_containers.h\"")

    if(FAISCA_CT_DIR)
        add_executable(bench_loader_ct ${FAISCA_LOADER_BENCH_SOURCES})
        target_include_directories(bench_loader_ct PRIVATE include bench/adapters ${FAISCA_CT_DIR}/include)
        target_compile_definitions(bench_loader_ct PRIVATE
            "FAISCA_BENCH_NAME=\"ct containers\""
            "FAISCA_BENCH_CONTAINERS_HEADER=\"ct_containers.h\"")

        add_executable(bench_containers bench/bench_containers.cpp)
        target_link_libraries(bench_containers PRIVATE faisca_lib)
        target_include_directories(bench_containers PRIVATE ${FAISCA_CT_DIR}/include)
    else()
        message(STATUS "faisca: FAISCA_CT_DIR not set, the ct benchmarks are disabled")
    endif()

    if(FAISCA_LLAMA_DIR)
        set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
        set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
        add_subdirectory(${FAISCA_LLAMA_DIR} ${CMAKE_BINARY_DIR}/llama EXCLUDE_FROM_ALL)

        add_executable(compare_ggml bench/compare_ggml.cpp)
        target_link_libraries(compare_ggml PRIVATE faisca_lib ggml-base)
        target_include_directories(compare_ggml PRIVATE ${FAISCA_LLAMA_DIR}/ggml/include)

        add_executable(bench_tokenizer bench/bench_tokenizer.cpp)
        target_link_libraries(bench_tokenizer PRIVATE faisca_lib llama)
        target_include_directories(bench_tokenizer PRIVATE ${FAISCA_LLAMA_DIR}/include ${FAISCA_LLAMA_DIR}/ggml/include)

        add_executable(llama_tokenizer_dump tests/reference/llama_tokenizer_dump.cpp)
        target_link_libraries(llama_tokenizer_dump PRIVATE llama)
        target_include_directories(llama_tokenizer_dump PRIVATE ${FAISCA_LLAMA_DIR}/include ${FAISCA_LLAMA_DIR}/ggml/include)

        add_executable(llama_layer_dump tests/reference/llama_layer_dump.cpp)
        target_link_libraries(llama_layer_dump PRIVATE llama)
        target_include_directories(llama_layer_dump PRIVATE ${FAISCA_LLAMA_DIR}/include ${FAISCA_LLAMA_DIR}/ggml/include)
    endif()
    if(FAISCA_WHISPER_DIR AND FAISCA_WHISPER_BUILD)
        add_executable(whisper_layer_dump tests/reference/whisper_layer_dump.cpp)
        target_include_directories(whisper_layer_dump PRIVATE ${FAISCA_WHISPER_DIR}/include ${FAISCA_WHISPER_DIR}/ggml/include)
        target_link_directories(whisper_layer_dump PRIVATE ${FAISCA_WHISPER_BUILD}/bin)
        target_link_libraries(whisper_layer_dump PRIVATE whisper ggml ggml-base)
        set_target_properties(whisper_layer_dump PROPERTIES BUILD_RPATH ${FAISCA_WHISPER_BUILD}/bin)
    endif()
endif()
