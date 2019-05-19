cmake_minimum_required(VERSION 3.10)
project(pik_simd LANGUAGES CXX)

file(GLOB pik_simd_srcs
	pik/simd/arch.h
	pik/simd/arm64_neon.h
	pik/simd/compiler_specific.h
	pik/simd/foreach_target.h
	#pik/simd/instruction_matrix.pdf
	#pik/simd/LICENSE
	#pik/simd/Makefile
	#pik/simd/quick_reference.md
	#pik/simd/README.md
	pik/simd/scalar.h
	pik/simd/shared.h
	pik/simd/simd.h
	#pik/simd/simd_test.cc
	#pik/simd/simd_test.cctest
	pik/simd/targets.cc
	pik/simd/targets.h
	pik/simd/util.h
	pik/simd/x86_avx2.h
	pik/simd/x86_sse4.h
)

add_library(pik_simd STATIC ${pik_simd_srcs})
target_include_directories( pik_simd PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" )
#target_link_libraries( pik_simd boost_variant )
