from setuptools import setup, find_packages
 
setup(
    name='hazardmapper',                # Name of your package
    version='0.1.0',                    # Version
    packages=find_packages(),           # Automatically find all packages in hazardmapper/
    install_requires=[
        # List your dependencis here (versions optional)
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'torch',
        'wandb',
        # add more if needed
    ],
    include_package_data=True,          # Include package data specified in MANIFEST.in (if any)
    author='Jeremy Palmerio',                 # Optional: your name
    description='Master Thesis Project', # Optional: description
)