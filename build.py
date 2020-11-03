import os
import sys
from typing import Any, Dict

from setuptools import Extension
from setuptools.command.build_ext import build_ext

# See example project: https://github.com/pybind/python_example
#
# Modifications include:
# - compiling against ortools headers
# - linking ortools dynamic libraries
# - setting the @rpath during the linking phase (Only tested on macos)


ORTOOLS_DIR = os.environ.get("ORTOOLS_DIR", "")


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __str__(self):
        import pybind11
        return pybind11.get_include()


ext_modules = [
    Extension(
        'ember.template._native.embed',
        ["ember/template/_native/embed.cpp"],
        include_dirs=[get_pybind_include(), os.path.join(ORTOOLS_DIR, "include")],
        library_dirs=[os.path.join(ORTOOLS_DIR, "lib")],
        libraries=["protobuf", "glog", "gflags", "ortools"],
        # extra_compile_args=["-Llib", "-Llib64"],
        extra_link_args=[f"-Wl,-rpath,{os.path.join(ORTOOLS_DIR, 'lib')}"],
        language='c++'
    ),
]


# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    from distutils.errors import CompileError

    extra = ["-stdlib=libc++"] if sys.platform == "darwin" else []

    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname] + extra)
        except CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.
    The newer version is prefered over c++11 (when it is available).
    """
    flags = ['-std=c++17', '-std=c++14', '-std=c++11']

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError('Unsupported compiler -- at least C++11 support '
                       'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == 'unix':
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')

        for ext in self.extensions:
            ext.define_macros = [
                ('VERSION_INFO', '"{}"'.format(self.distribution.get_version()))]
            ext.extra_compile_args += opts
            ext.extra_link_args += link_opts
        build_ext.build_extensions(self)


def build(setup_kwargs: Dict[str, Any]) -> None:
    if ORTOOLS_DIR:
        setup_kwargs.update(
            {
                "ext_modules": ext_modules,
                "cmdclass": dict(build_ext=BuildExt),
                "zip_safe": False,
            }
        )
    else:
        print(f"""
            Warning: ORTOOLS_DIR is not exported. 
            Therefore, the solver will fallback to the python API. 
            Note that this will incur a performance penalty when loading the model
        """, file=sys.stderr)
