import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", 'r') as fh:
    requirements = fh.read().split('\n')

setuptools.setup(
    name="rps-databases",  # Replace with your own username
    version="0.0.9",
    author="Wilian Silva",
    author_email="wilianzilv@gmail.com",
    description="Funções para conexão com banco de dados",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rpscapital/rps-databases",
    packages=setuptools.find_packages(),
    python_requires='>=3.5.2',
    install_requires=requirements
)
