from setuptools import setup, find_packages

setup(
    name="stackforge",
    version="0.0.1",
    description="A Python package for stacking and analyzing astrophysical profiles from IllustrisTNG simulations.",
    author="Javier Urrutia",
    author_email="javierul@stanford.edu",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "h5py",
        "astropy",
        "scipy"
    ],
    include_package_data=True,
    url="https://github.com/javierurrutia/stackforge",
)