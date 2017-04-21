HOST=lin@iccluster130.iccluster.epfl.ch
HOST=lin@iccluster036.iccluster.epfl.ch

build_platform:
	sh build_platform.sh

ssh_server:
	ssh ${HOST}

cp_all_gpu:
	rsync -av -e ssh ../test ${HOST}:~/
