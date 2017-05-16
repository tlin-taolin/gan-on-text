HOST=lin@iccluster036.iccluster.epfl.ch

ssh_server:
	ssh ${HOST}

ssh_tensorboard:
	ssh -L 16006:127.0.0.1:6006 ${HOST}

ssh_notebook:
	ssh -L 18888:localhost:8887 ${HOST}

cp_code_gpu:
	rsync -av -e ssh w5 ${HOST}:~/gan/code
