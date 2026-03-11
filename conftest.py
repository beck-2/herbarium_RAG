"""
Root pytest configuration.

Library conflict note (macOS — already fixed):
    torch and faiss-cpu each bundle their own copy of LLVM's libomp.dylib.
    Loading both in the same process causes an unconditional abort.
    KMP_DUPLICATE_LIB_OK=TRUE has no effect — that flag is Intel OpenMP only.

    Fix applied: faiss's libomp.dylib is symlinked to torch's libomp.dylib so
    only one copy is ever initialized:

        ln -sf <site-packages>/torch/lib/libomp.dylib \\
               <site-packages>/faiss/.dylibs/libomp.dylib

    If upgrading faiss-cpu re-installs the file, re-run the symlink command.
    The backup of the original is at faiss/.dylibs/libomp.dylib.bak.
"""
