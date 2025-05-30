cmake_minimum_required(VERSION 3.26)
project(onnxruntime_static_lib)

set(onnxruntime_BUILD_SHARED_LIB ON)

add_definitions(-D_SILENCE_ALL_CXX23_DEPRECATION_WARNINGS)

add_subdirectory(${ONNXRUNTIME_SOURCE_DIR}/cmake onnxruntime EXCLUDE_FROM_ALL)

function(bundle_static_library bundled_target_name)
    set(extension_blocklist
        .dylib
        .so
        .dll
        .tbd
    )

    function(recursively_collect_dependencies input_target)
        if(NOT TARGET ${input_target})
            if(EXISTS ${input_target})
                get_filename_component(extension "${input_target}" EXT)

                if(NOT extension IN_LIST extension_blocklist)
                    list(APPEND static_libs "${input_target}")
                    set(static_libs ${static_libs} PARENT_SCOPE)
                endif()
            endif()

            return()
        endif()

        get_target_property(alias ${input_target} ALIASED_TARGET)

        if(TARGET ${alias})
            set(input_target ${alias})
        endif()

        get_property(library_already_added GLOBAL PROPERTY ${target_name}_static_bundle_${input_target})

        if(library_already_added)
            return()
        endif()

        set_property(GLOBAL PROPERTY ${target_name}_static_bundle_${input_target} ON)

        get_target_property(input_type ${input_target} TYPE)

        if(${input_type} STREQUAL "STATIC_LIBRARY")
            list(APPEND static_libs "$<TARGET_FILE:${input_target}>")
            get_target_property(dependencies ${input_target} LINK_LIBRARIES)
        elseif(${input_type} STREQUAL "INTERFACE_LIBRARY")
            get_target_property(dependencies ${input_target} INTERFACE_LINK_LIBRARIES)
        else()
            get_target_property(dependencies ${input_target} LINK_LIBRARIES)
        endif()

        if(dependencies)
            foreach(dependency IN LISTS dependencies)
                recursively_collect_dependencies(${dependency})
            endforeach()
        endif()

        set(static_libs ${static_libs} PARENT_SCOPE)
    endfunction()

    foreach(target_name IN ITEMS ${ARGN})
        recursively_collect_dependencies(${target_name})
    endforeach()

    list(REMOVE_DUPLICATES static_libs)
    set(static_libs ${static_libs} PARENT_SCOPE)

    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/static_libs.txt.in)

    foreach(static_lib IN LISTS static_libs)
        file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/static_libs.txt.in "${static_lib}\n")
    endforeach()

    file(GENERATE
        OUTPUT static_libs.txt
        INPUT ${CMAKE_CURRENT_BINARY_DIR}/static_libs.txt.in)

    set(bundled_target_full_name
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${bundled_target_name}${CMAKE_STATIC_LIBRARY_SUFFIX})

    if(MSVC)
        set(lib ${CMAKE_AR})
        add_custom_command(
            COMMAND ${lib} /NOLOGO /OUT:${bundled_target_full_name} ${static_libs}
            OUTPUT ${bundled_target_full_name}
            COMMENT "Bundling ${bundled_target_name}"
            VERBATIM)
    elseif(APPLE)
        find_program(libtool libtool)
        add_custom_command(
            COMMAND ${libtool} -static -o ${bundled_target_full_name} ${static_libs}
            OUTPUT ${bundled_target_full_name}
            COMMENT "Bundling ${bundled_target_name}"
            VERBATIM)
    else()
        file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${bundled_target_name}.ar.in
            "CREATE ${bundled_target_full_name}\n")

        foreach(static_lib IN LISTS static_libs)
            file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/${bundled_target_name}.ar.in
                "ADDLIB ${static_lib}\n")
        endforeach()

        file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/${bundled_target_name}.ar.in "SAVE\n")
        file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/${bundled_target_name}.ar.in "END\n")

        file(GENERATE
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${bundled_target_name}.ar
            INPUT ${CMAKE_CURRENT_BINARY_DIR}/${bundled_target_name}.ar.in)

        set(ar ${CMAKE_AR})

        if(CMAKE_INTERPROCEDURAL_OPTIMIZATION)
            set(ar ${CMAKE_CXX_COMPILER_AR})
        endif()

        add_custom_command(
            COMMAND ${ar} -M < ${CMAKE_CURRENT_BINARY_DIR}/${bundled_target_name}.ar
            OUTPUT ${bundled_target_full_name}
            COMMENT "Bundling ${bundled_target_name}"
            VERBATIM)
    endif()

    add_custom_target(bundling_${bundled_target_name} ALL DEPENDS ${bundled_target_full_name})

    foreach(target_name IN ITEMS ${ARGN})
        if(TARGET ${target_name})
            add_dependencies(bundling_${bundled_target_name} ${target_name})
        endif()
    endforeach()

    add_library(${bundled_target_name} STATIC IMPORTED GLOBAL)
    set_target_properties(${bundled_target_name}
        PROPERTIES
        IMPORTED_LOCATION ${bundled_target_full_name})

    foreach(target_name IN ITEMS ${ARGN})
        set_property(TARGET ${bundled_target_name} APPEND
            PROPERTY INTERFACE_INCLUDE_DIRECTORIES $<TARGET_PROPERTY:${target_name},INTERFACE_INCLUDE_DIRECTORIES>)
        set_property(TARGET ${bundled_target_name} APPEND
            PROPERTY INTERFACE_COMPILE_DEFINITIONS $<TARGET_PROPERTY:${target_name},INTERFACE_COMPILE_DEFINITIONS>)
    endforeach()

    add_dependencies(${bundled_target_name} bundling_${bundled_target_name})
endfunction()
bundle_static_library(${PROJECT_NAME} onnxruntime)

if(onnxruntime_USE_CUDA)
    add_dependencies(onnxruntime onnxruntime_providers_cuda)
    install(IMPORTED_RUNTIME_ARTIFACTS onnxruntime_providers_shared DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(IMPORTED_RUNTIME_ARTIFACTS onnxruntime_providers_cuda DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
if(onnxruntime_USE_TENSORRT)
    add_dependencies(onnxruntime onnxruntime_providers_tensorrt)
    install(IMPORTED_RUNTIME_ARTIFACTS onnxruntime_providers_shared DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(IMPORTED_RUNTIME_ARTIFACTS onnxruntime_providers_tensorrt DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
if(onnxruntime_USE_NV)
    add_dependencies(onnxruntime onnxruntime_providers_nv_tensorrt_rtx)
    install(IMPORTED_RUNTIME_ARTIFACTS onnxruntime_providers_shared DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(IMPORTED_RUNTIME_ARTIFACTS onnxruntime_providers_nv_tensorrt_rtx DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
if(onnxruntime_USE_DNNL)
    add_dependencies(onnxruntime onnxruntime_providers_dnnl)
    install(IMPORTED_RUNTIME_ARTIFACTS onnxruntime_providers_shared DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(IMPORTED_RUNTIME_ARTIFACTS onnxruntime_providers_dnnl DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
if(onnxruntime_USE_ROCM)
    add_dependencies(onnxruntime onnxruntime_providers_rocm)
    install(IMPORTED_RUNTIME_ARTIFACTS onnxruntime_providers_shared DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(IMPORTED_RUNTIME_ARTIFACTS onnxruntime_providers_rocm DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
if(onnxruntime_USE_WEBGPU)
    # both of these are required for some reason, LIBRARY RUNTIME doesn't work and it still copies to /bin, but at least it
    # *also* copies to /lib with this
    install(TARGETS webgpu_dawn LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(IMPORTED_RUNTIME_ARTIFACTS webgpu_dawn DESTINATION ${CMAKE_INSTALL_LIBDIR})
    if(WIN32)
        install(FILES "$<TARGET_FILE_DIR:dxcompiler>/dxil.dll" "$<TARGET_FILE_DIR:dxcompiler>/dxcompiler.dll" TYPE LIB)
    endif()
endif()
if(onnxruntime_USE_DML)
    install(FILES "$<TARGET_FILE_DIR:onnxruntime_providers_dml>/DirectML.dll" TYPE LIB)
endif()

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${PROJECT_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}
    RENAME ${CMAKE_STATIC_LIBRARY_PREFIX}onnxruntime${CMAKE_STATIC_LIBRARY_SUFFIX}
    TYPE LIB
)
