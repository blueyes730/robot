from setuptools import find_packages, setup

package_name = "localization"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="sgupta",
    maintainer_email="sgupta@users.noreply.github.com",
    description="Consume standard inertial measurements for localization.",
    license="Apache-2.0",
    entry_points={"console_scripts": ['localization_node = localization.localization_node:main']},
)
