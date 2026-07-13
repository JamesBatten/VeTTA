from setuptools import find_packages
from setuptools import setup

setup(
    name="vetta",
    author="James Batten",
    version="0.0.1",
    description="Vector representations of vessel trees",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pydantic-settings",
        "torch",
        "scipy",
    ],
    extras_require={
        "training": [
            "pyzmq",
            "opencv-python-headless",
            "matplotlib",
            "Pillow",
        ],
    },
)
