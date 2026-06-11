import importlib

setuptools_spec = importlib.util.find_spec('setuptools')
if setuptools_spec is not None:
    setuptools = importlib.import_module('setuptools')
    setup = setuptools.setup
    find_packages = setuptools.find_packages
else:
    distutils = importlib.import_module('distutils.core')
    setup = distutils.setup

    def find_packages():
        return [
            'src',
        ]

setup(
    name='medical_chatbot',
    version='0.1.0',
    author='Tanisha',
    author_email='me.tanisha@example.com',
    packages=find_packages(),
    install_requires=[]
)