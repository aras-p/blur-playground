cmake_minimum_required (VERSION 3.21)
if(POLICY CMP0091)
    cmake_policy(SET CMP0091 NEW) # enable MSVC_RUNTIME_LIBRARY
endif()

project ("blur_playground")

function(add_sokol_shader target_name shader_file)
    set(SOKOL_SHDC "${CMAKE_SOURCE_DIR}/tools/sokol-shdc")
    set(SHADER_OUTPUT "${CMAKE_SOURCE_DIR}/${shader_file}.h")
    add_custom_command(
        OUTPUT ${SHADER_OUTPUT}
        COMMAND ${SOKOL_SHDC}
            --input ${CMAKE_SOURCE_DIR}/${shader_file}
            --output ${SHADER_OUTPUT}
            --slang glsl430:hlsl5:metal_macos
        DEPENDS ${CMAKE_SOURCE_DIR}/${shader_file}
        COMMENT "Compiling shader ${shader_file} with sokol-shdc"
        VERBATIM
    )
    target_sources(${target_name} PRIVATE ${SHADER_OUTPUT})
endfunction()

add_executable (blur_playground
	src/app.cpp
    src/blurs.cpp
    src/blurs.h
    src/lib_impl.cpp
    src/texture.h
    readme.md
    shaders/blur.glsl
    shaders/calc_err.glsl
    shaders/display_tex.glsl
    lib/imgui/imgui.cpp
    lib/imgui/imgui_draw.cpp
    lib/imgui/imgui_tables.cpp
    lib/imgui/imgui_widgets.cpp
    lib/tinyexr/tinyexr.cc
    lib/tinyexr/miniz/miniz.c
)

set(SHADER_FILES
    shaders/blur.glsl
    shaders/calc_err.glsl
    shaders/display_tex.glsl
)
foreach(shader ${SHADER_FILES})
    add_sokol_shader(blur_playground ${shader})
endforeach()

target_include_directories(blur_playground PRIVATE lib/tinyexr/miniz)

set_property(TARGET blur_playground PROPERTY CXX_STANDARD 17)
set_property(TARGET blur_playground PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
target_compile_definitions(blur_playground PRIVATE _CRT_SECURE_NO_DEPRECATE _CRT_NONSTDC_NO_WARNINGS NOMINMAX)

if (APPLE)
	target_compile_options(blur_playground PRIVATE -x objective-c++)
	target_link_libraries(blur_playground PRIVATE
		"-framework Cocoa"
		"-framework Metal"
		"-framework MetalKit"
		"-framework QuartzCore"
	)
endif ()
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
	target_link_libraries(blur_playground PRIVATE X11 Xi Xcursor GL)
endif ()
