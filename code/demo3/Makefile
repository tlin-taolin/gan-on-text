HOST=lin@iccluster130.iccluster.epfl.ch
HOST=lin@iccluster036.iccluster.epfl.ch

build_package:
	python setup.py install

get_data:
	rsync -av -e ssh ${HOST}:~/gan/code/demo3/data/training/runs data/archive/
