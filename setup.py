"""Installation script for the 'dexhand' python package."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from setuptools import setup, find_packages

import os

root_dir = os.path.dirname(os.path.realpath(__file__))


# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # RL
    "gym==0.23.1",
    "torch",
    "omegaconf",
    "termcolor",
    "jinja2",
    "hydra-core>=1.2",
    "rl-games>=1.6.0",
    "tensorboard",  # For training metrics logging
    "pyvirtualdisplay",
    "urdfpy==0.0.22",
    "pysdf==0.1.9",
    "warp-lang==0.10.1",
    "trimesh==3.23.5",
]

# Optional dependencies for additional features
EXTRAS_REQUIRE = {
    "streaming": ["flask>=2.0.0"],  # For HTTP video streaming
    "video": ["opencv-python>=4.5.0"],  # For video recording
    "all": ["flask>=2.0.0", "opencv-python>=4.5.0"],  # All optional features
}


# Installation operation
setup(
    name="dexhand_env",
    author="DexRobot Inc.",
    version="0.1.0",
    description="Reinforcement learning environment for dexterous manipulation with robotic hands",
    keywords=["robotics", "rl", "dexterous", "manipulation"],
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    packages=find_packages("."),
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6, 3.7, 3.8",
    ],
    zip_safe=False,
)

# EOF
