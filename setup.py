import setuptools

with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fermatrica-rep"
    , version="0.2"
    , author="OKKAM CS: AMeshkov, Ivan Kocherov, Grigoriy Selyavin, Anton Sevastianov, Anna Gladysheva, Nargiz Bagirova"
    , description="FERMATRICA flexible econometrics framework: Reporting"
    , long_description=long_description
    , long_description_content_type="text/markdown"
    , url="https://github.com/FERMATRICA/fermatrica_rep"
    , packages=setuptools.find_packages(exclude=['samples*'])
    , package_data={'': ['*.csv', '*.sql', '*.xlsx', '*.pptx', '*.html', '*.xml', '*.txt']}
    # , include_package_data=True
    , classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
    , python_requires=">=3.10"
)
