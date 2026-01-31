from importlib import resources as importlib_resources
import pyopencl as cl
import os



class Kernel:
    def __init__(self, path: str, targets: list[str]):
        self.path = path
        self.__targets = targets

        self.program: cl.Program | None = None
        self.methods = {}

    def __getattr__(self, target: str):
        if target in self.methods:
            return self.methods[target]

        raise AttributeError(f"Kernel '{self.path}' has no method '{target}'")

    def build(self):
        for target in self.__targets:
            try:
                method = cl.Kernel(self.program, target)
            except cl.RepeatedKernelRetrieval:
                raise NotImplementedError(f"[Kernel] Method '{target}' is not defined in kernel '{self.path}'")

            if not method:
                raise NotImplementedError(f"Kernel '{self.path}' has no method '{target}'")

            self.methods[target] = method


class KernelBuilder:
    def __init__(self, context):
        self.context = context
        self.known_programs = {}

    def build_from_path(self, path: str, silent: bool) -> cl.Program:
        path = os.path.normpath(os.path.join(
            str(importlib_resources.files("pymyconet2")), "kernels", path
        ))

        if path not in self.known_programs:
            if not silent:
                print(f"[Kernel] Building kernel from source '{path}'")
            with open(path, "r") as f:
                prg = cl.Program(self.context, f.read()).build()

            self.known_programs[path] = prg
            return prg

        return self.known_programs[path]

    def build_kernel(self, kernel: Kernel, silent) -> None:
        kernel.program = self.build_from_path(kernel.path, silent)
        kernel.build()