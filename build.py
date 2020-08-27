from typing import Dict, Any


def build(setup_kwargs):
    print("Did something")
    # setup_kwargs.update(
    #     {
    #         "ext_modules": ext_modules,
    #         "cmdclass": dict(build_ext=ExtensionBuilder),
    #         "zip_safe": False,
    #     }
    # )
    return setup_kwargs


if __name__ == "__main__":
    build()
