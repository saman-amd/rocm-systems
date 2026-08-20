# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# The guest-visible clock, alone in its own object library because its readers
# are not the timing core's readers: the KFD ioctl handlers, the command
# processor and the generated ISA execute bodies all read it, and those end up
# in images -- the DBT translation tool, the HSA hook library -- that have no
# business linking a model loader, a config parser or FlatBuffers. It depends on
# nothing but the standard library, so keeping it separate costs one target and
# keeps that dependency edge from existing.
rj_add_object_library(
    rocjitsu_timing_clock
    simulated_clock.cpp
)

# Timing core: the `timing` config block and its TimingHost, the observation
# layer that turns execution hooks into timing events, and the host-side loader
# that dlopen()s a model. Linked into the main shared library, the CLI and the
# unit-test binary.
rj_add_object_library(
    rocjitsu_timing
    timing_config.cpp
    timing_loader.cpp
    observer.cpp
)
# timing_config.cpp and timing_loader.cpp parse config with FlexBuffers and
# reuse the plugin loader's schema resolver, whose symbols live in
# rocjitsu_plugin_loader; every final image that links this one already links
# that one. timing_loader.cpp resolves models with dlopen/dlsym via
# util/dynamic_loader.h, so it needs libdl on glibc < 2.34 where dl* has not yet
# moved into libc; PUBLIC propagates that to consumers and ${CMAKE_DL_LIBS} is
# empty on newer glibc, so it stays portable.
target_link_libraries(
    rocjitsu_timing
    PRIVATE flatbuffers
    PUBLIC ${CMAKE_DL_LIBS}
)
add_dependencies(rocjitsu_timing flatbuffers_schemas)

# Model logic as an object library, with no loader exports in it: it is linked
# both into the runtime model shared object below and into the unit-test binary,
# where two models' exports would collide. The exports live in the separate
# models/<name>/model_export.cpp, compiled only into the .so.
rj_add_object_library(
    rocjitsu_timing_leaky
    models/leaky/model.cpp
)

# Runtime timing-model shared objects are only meaningful on Linux (the
# interposer platform). Each is a `librocjitsu_timing_<name>.so` carrying only
# its own model logic plus the three loader entry points, restricted to be the
# only exported symbols by a linker version script. It is self-contained (any
# simulator code it references is statically bundled) and is not linked against
# librocjitsu.so, so loading it never pulls the interposer into the host
# process.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Emit models beside the interposer and the execution plugins (librocjitsu.so
    # is emitted at the top of the build tree) so the loader's search finds them
    # by soname, and so the build tree matches the installed layout where
    # models, plugins and the interposer share <libdir>.
    set(RJ_TIMING_OUTPUT_DIR ${PROJECT_BINARY_DIR})

    # rj_add_timing_model_so(<name> <object_lib> <export_source>)
    function(rj_add_timing_model_so name object_lib export_src)
        set(target rocjitsu_timing_${name}_so)
        add_library(${target} MODULE ${export_src})
        target_include_directories(
            ${target}
            PRIVATE
                ${ROCJITSU_INCLUDE_DIR}
                ${ROCJITSU_SRC_DIR}
                ${HSA_INCLUDE_DIR}
                # The model's object library links util/simdojo PRIVATE, so pull
                # in their public header paths for the export source's compile.
                $<TARGET_PROPERTY:util,INTERFACE_INCLUDE_DIRECTORIES>
                $<TARGET_PROPERTY:simdojo,INTERFACE_INCLUDE_DIRECTORIES>
        )
        target_link_libraries(${target} PRIVATE ${object_lib})
        add_dependencies(${target} flatbuffers_schemas)
        # Hide everything except the loader exports, which carry an explicit
        # default-visibility attribute via ROCJITSU_DEFINE_TIMING_MODEL.
        set_target_properties(
            ${target}
            PROPERTIES
                OUTPUT_NAME rocjitsu_timing_${name}
                LIBRARY_OUTPUT_DIRECTORY ${RJ_TIMING_OUTPUT_DIR}
                CXX_VISIBILITY_PRESET hidden
                C_VISIBILITY_PRESET hidden
                VISIBILITY_INLINES_HIDDEN ON
        )
        target_link_options(
            ${target}
            PRIVATE
                "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/timing_export.map"
        )
        set_property(
            TARGET ${target}
            APPEND
            PROPERTY LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/timing_export.map
        )
        install(TARGETS ${target} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
    endfunction()

    rj_add_timing_model_so(leaky rocjitsu_timing_leaky
        ${CMAKE_CURRENT_SOURCE_DIR}/models/leaky/model_export.cpp
    )
endif()
