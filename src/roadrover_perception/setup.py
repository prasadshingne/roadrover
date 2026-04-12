import os
from glob import glob
from setuptools import setup

package_name = 'roadrover_perception'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pss',
    maintainer_email='pss@todo.todo',
    description='Perception pipeline for roadrover',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_preprocessor = roadrover_perception.image_preprocessor:main',
            'object_detector = roadrover_perception.object_detector:main',
        ],
    },
)
