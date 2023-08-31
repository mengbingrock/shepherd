import sys
import os
import ctypes
from ctypes import (
    c_double,
    c_int,
    c_float,
    c_char_p,
    c_int32,
    c_uint32,
    c_void_p,
    c_bool,
    POINTER,
    _Pointer,  # type: ignore
    Structure,
    Array,
    c_uint8,
    c_size_t,
)
import pathlib
from typing import List, Union




# Load the library
def _load_shared_library(lib_base_name: str):
    # Construct the paths to the possible shared library names
    _base_path = pathlib.Path(__file__).parent.resolve()
    # Searching for the library in the current directory under the name "libllama" (default name
    # for llamacpp) and "llama" (default name for this repo)
    _lib_paths: List[pathlib.Path] = []
    # Determine the file extension based on the platform
    if sys.platform.startswith("linux"):
        _lib_paths += [
            _base_path / f"lib{lib_base_name}.so",
        ]
    elif sys.platform == "darwin":
        _lib_paths += [
            _base_path / f"lib{lib_base_name}.so",
            _base_path / f"lib{lib_base_name}.dylib",
        ]
    elif sys.platform == "win32":
        _lib_paths += [
            _base_path / f"{lib_base_name}.dll",
        ]
    else:
        raise RuntimeError("Unsupported platform")

    cdll_args = dict()  # type: ignore
    # Add the library directory to the DLL search path on Windows (if needed)


    # Try to load the shared library, handling potential errors
    for _lib_path in _lib_paths:
        print("_lib_path = ", _lib_path)
        if _lib_path.exists():
            try:
                return ctypes.CDLL(str(_lib_path), **cdll_args)
            except Exception as e:
                raise RuntimeError(f"Failed to load shared library '{_lib_path}': {e}")

    raise FileNotFoundError(
        f"Shared library with base name '{lib_base_name}' not found"
    )



# Specify the base name of the shared library to load
_lib_base_name = "model"

# Load the library
_lib = _load_shared_library(_lib_base_name)


# LLAMA_API struct llama_context_params llama_context_default_params();
def inference(argc: c_int, argv: c_char_p):
    return _lib.inference(argc, argv)


#_lib.inference.argtypes = [c_int, c_char_p]
#_lib.inference.restype = c_int

if __name__ == "__main__":
    inference(2 ,bytes( "stories15M.bin", encoding = 'utf-8'))


