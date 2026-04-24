/**
 * Compile LLVM IR to LTOIR using libNVVM
 *
 * This tool takes cuda-oxide generated .ll files and compiles them to LTOIR
 * using libNVVM with the -gen-lto flag.
 *
 * Build:
 *   gcc -o compile_ltoir compile_ltoir.c \
 *       -I/usr/local/cuda/nvvm/include \
 *       -L/usr/local/cuda/nvvm/lib64 -lnvvm \
 *       -Wl,-rpath,/usr/local/cuda/nvvm/lib64
 *
 * Usage:
 *   ./compile_ltoir <input.ll> <arch> [output.ltoir]
 *
 * Example:
 *   ./compile_ltoir device_ffi_test.ll sm_120 device_ffi_test.ltoir
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <nvvm.h>

/**
 * Check an NVVM result and exit with error message if it failed.
 *
 * @param result  The nvvmResult to check
 * @param msg     Context message for the error
 * @param prog    Optional program handle for retrieving compilation log
 */
static void check_nvvm(nvvmResult result, const char* msg, nvvmProgram prog) {
    if (result != NVVM_SUCCESS) {
        fprintf(stderr, "Error: %s - %s\n", msg, nvvmGetErrorString(result));
        if (prog) {
            size_t logSize;
            if (nvvmGetProgramLogSize(prog, &logSize) == NVVM_SUCCESS && logSize > 1) {
                char* log = malloc(logSize);
                if (nvvmGetProgramLog(prog, log) == NVVM_SUCCESS) {
                    fprintf(stderr, "Log:\n%s\n", log);
                }
                free(log);
            }
        }
        exit(1);
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input.ll> <arch> [output.ltoir]\n", argv[0]);
        fprintf(stderr, "  arch: sm_100, sm_120, etc.\n");
        fprintf(stderr, "\nExample:\n");
        fprintf(stderr, "  %s device_ffi_test.ll sm_120 device_ffi_test.ltoir\n", argv[0]);
        return 1;
    }

    const char* inputFile = argv[1];
    const char* arch = argv[2];
    const char* outputFile = argc > 3 ? argv[3] : NULL;

    // Print libNVVM version info
    int major, minor;
    nvvmVersion(&major, &minor);
    printf("libNVVM version: %d.%d\n", major, minor);

    int irMajor, irMinor, dbgMajor, dbgMinor;
    nvvmIRVersion(&irMajor, &irMinor, &dbgMajor, &dbgMinor);
    printf("NVVM IR version: %d.%d (debug: %d.%d)\n", irMajor, irMinor, dbgMajor, dbgMinor);

    // Convert sm_XXX to compute_XXX
    char archOpt[64];
    snprintf(archOpt, sizeof(archOpt), "compute_%s", arch + 3);
    printf("Target architecture: %s\n", archOpt);

    // Read input file
    FILE* f = fopen(inputFile, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open %s\n", inputFile);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buffer = malloc(size + 1);
    fread(buffer, 1, size, f);
    buffer[size] = '\0';
    fclose(f);
    printf("Read %zu bytes from %s\n", size, inputFile);

    // Create program
    nvvmProgram prog;
    check_nvvm(nvvmCreateProgram(&prog), "nvvmCreateProgram", NULL);

    // Add module
    nvvmResult addResult = nvvmAddModuleToProgram(prog, buffer, size, inputFile);
    if (addResult != NVVM_SUCCESS) {
        fprintf(stderr, "nvvmAddModuleToProgram failed: %s\n", nvvmGetErrorString(addResult));
        size_t logSize;
        if (nvvmGetProgramLogSize(prog, &logSize) == NVVM_SUCCESS && logSize > 1) {
            char* log = malloc(logSize);
            if (nvvmGetProgramLog(prog, log) == NVVM_SUCCESS) {
                fprintf(stderr, "Log:\n%s\n", log);
            }
            free(log);
        }
        nvvmDestroyProgram(&prog);
        free(buffer);
        return 1;
    }
    printf("Module added successfully\n");

    // Compile options - CRITICAL: -gen-lto generates LTOIR
    char archOption[128];
    snprintf(archOption, sizeof(archOption), "-arch=%s", archOpt);

    const char* options[] = {
        archOption,
        "-gen-lto"  // Generate LTOIR for link-time optimization
    };
    int numOptions = 2;

    printf("Compiling with options: %s %s\n", options[0], options[1]);

    // Compile
    nvvmResult compileResult = nvvmCompileProgram(prog, numOptions, options);
    if (compileResult != NVVM_SUCCESS) {
        fprintf(stderr, "nvvmCompileProgram failed: %s\n", nvvmGetErrorString(compileResult));
        size_t logSize;
        if (nvvmGetProgramLogSize(prog, &logSize) == NVVM_SUCCESS && logSize > 1) {
            char* log = malloc(logSize);
            if (nvvmGetProgramLog(prog, log) == NVVM_SUCCESS) {
                fprintf(stderr, "Log:\n%s\n", log);
            }
            free(log);
        }
        nvvmDestroyProgram(&prog);
        free(buffer);
        return 1;
    }
    printf("Compilation successful!\n");

    // Get result
    size_t resultSize;
    check_nvvm(nvvmGetCompiledResultSize(prog, &resultSize), "nvvmGetCompiledResultSize", prog);
    printf("LTOIR size: %zu bytes\n", resultSize);

    char* result = malloc(resultSize);
    check_nvvm(nvvmGetCompiledResult(prog, result), "nvvmGetCompiledResult", prog);

    // Save LTOIR
    if (outputFile) {
        FILE* out = fopen(outputFile, "wb");
        if (out) {
            fwrite(result, 1, resultSize, out);
            fclose(out);
            printf("Saved LTOIR to: %s\n", outputFile);
        } else {
            fprintf(stderr, "Error: Cannot write to %s\n", outputFile);
        }
    } else {
        // Generate output filename from input
        char autoOutput[256];
        snprintf(autoOutput, sizeof(autoOutput), "%.*s.ltoir",
                 (int)(strrchr(inputFile, '.') - inputFile), inputFile);
        FILE* out = fopen(autoOutput, "wb");
        if (out) {
            fwrite(result, 1, resultSize, out);
            fclose(out);
            printf("Saved LTOIR to: %s\n", autoOutput);
        }
    }

    // Cleanup
    free(result);
    free(buffer);
    nvvmDestroyProgram(&prog);

    printf("\n=== LLVM IR -> LTOIR compilation succeeded! ===\n");
    return 0;
}
