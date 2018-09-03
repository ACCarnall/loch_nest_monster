from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='lochnest_monster',

    version='0.1.2',

    description='An ancient and mysterious nested sampling algorithm',

    long_description=long_description,

    url='https://github.com/ACCarnall/lochnest_monster',

    author='Adam Carnall',

    author_email='adamc@roe.ac.uk',

    packages=["lochnest_monster", "lochnest_monster/bounds"],

    include_package_data=True,

    install_requires=["numpy>=1.14.2", "matplotlib>=2.2.2",
                      "scipy", "sklearn"],

    project_urls={
        "GitHub": "https://github.com/ACCarnall/lochnest_monster"
    }
)
