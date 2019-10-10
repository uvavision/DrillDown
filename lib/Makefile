all:
    # install external_tools locally
	python setup.py build_ext --inplace
	rm -rf build

install:
	# install external_tools to the Python site-packages
	python setup.py build_ext install
	rm -rf build