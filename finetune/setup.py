from setuptools import setup, find_packages

requirements = [
    "numpy",
    "scipy",
    "einops",
    "pyrender",
    "transformers",
    "omegaconf",
    "natsort",
    "cffi",
    "pandas",
    "tensorflow",
    "pyquaternion",
    "matplotlib",
    "bitsandbytes",
    "transforms3d",
    "clip @ git+https://github.com/openai/CLIP.git",
]

__version__ = "0.0.1"
setup(
    name="bridgevla",
    version=__version__,
    long_description="",
    packages=['bridgevla'],
    author='Peiyan Li',
    author_email='peiyan.li@cripac.ia.ac.cn',
    install_requires=requirements,
    extras_require={
        "xformers": [
            "xformers @ git+https://github.com/facebookresearch/xformers.git@main#egg=xformers",
        ]
    },
)
