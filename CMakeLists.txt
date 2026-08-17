cmake_minimum_required(VERSION 3.13)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project("evns-clouds-rtr" VERSION 0.0.1)

set(MANDRILL_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(Vulkan REQUIRED COMPONENTS glslc)
include_directories(${Vulkan_INCLUDE_DIRS})
link_libraries(${Vulkan_LIBRARIES})

set(MANDRILL_EXCLUDE_APPS ON)
include_directories(Mandrill/src)
include_directories(Mandrill/src/External)
include_directories(Mandrill/submodules/glm)
include_directories(src)

add_subdirectory(Mandrill)
add_subdirectory(src)

# Download resources
message(STATUS "Downloading EVNS of Clouds for RTR resources...")
include(FetchContent)
FetchContent_Declare(evns_resources
    URL https://fileadmin.cs.lth.se/graphics/research/papers/2026/environmental_volumetric_neural_shading_of_clouds_for_real_time_rendering/evns_clouds_rtr_resources.zip
    URL_HASH SHA256=41109f85ca07fe6f651039ad2e6869cbf1e861084510816c274e8356a1907d68
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/res"
    DOWNLOAD_EXTRACT_TIMESTAMP YES
)
FetchContent_MakeAvailable(evns_resources)
