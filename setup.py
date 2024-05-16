from setuptools import find_packages, setup

def get_requirements(file_path:str)->list[str]:
    """
    This function will return the list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        [req.replace("\n","") for req in requirements]

        if '-e .' in requirements:
            requirements.remove('-e .')
    
    return requirements


setup(
    name="KKBOX_Churn_Classifier",
    version='0.0.1',
    author='Jamie',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)


