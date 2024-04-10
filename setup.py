from setuptools import find_packages, setup
HYPEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> list:
    """
    This function will return the list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


def read_version(fname="src/version.py"):
    exec(compile(open(fname, encoding="utf-8").read(), fname, "exec"))
    return locals()["__version__"]


requirements = []


setup(
    name="end2endML",
    version=read_version(),
    author="Ying Liu",
    author_email="sophia.j.liu@gmail.com",
    # pckages=find_packages(),
    python_requires="==3.8.*",
    # install_requires=[
    #     str(r)
    #     for r in pkg_resources.parse_requirements(
    #         Path(__file__).with_name("requirements.txt").open()
    #     )
    # ],
    install_requires=get_requirements("requirements.txt"),
    description="End to End Machine Learning Project template",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    # readme="README.md",
    license="MIT",
)
