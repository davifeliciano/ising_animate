import setuptools

with open("README.md", "r", encoding="utf-8") as rm:
    long_description = rm.read()

setuptools.setup(
    name="ising_animate",
    version="0.0.2",
    author="Davi Feliciano",
    author_email="dfeliciano37@gmail.com",
    description="A module to easily generate animations of the Ising Model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/davifeliciano/ising_model",
    project_urls={
        "Bug Tracker": "https://github.com/davi_feliciano/ising_model/issues",
        "Documentation": "https://davifeliciano.github.io/ising_animate/index.html",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=["arrow", "matplotlib", "numpy", "progressbar2"],
    python_requires=">=3.6",
)
