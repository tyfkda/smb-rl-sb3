
.PHONY: all
all:
	@echo 'Specify target'
	@echo 'You must activate venv before running `make setup`'

.PHONY: setup
setup:	setup-gym-smb
	pip install -r requirements.txt

.PHONY: training
training:
	@echo 'Training: Try "python main.py [options]"'

.PHONY: replay
replay:
	@echo 'Replay: Try "python main.py --replay [options]"'

.PHONY: setup-gym-smb
setup-gym-smb:	make-nes-py
	sed -i.bak "s%^nes-py.*%$(shell pwd)/nes-py/dist/nes_py-8.2.1.tar.gz%" gym-super-mario-bros/requirements.txt
	rm -f gym-super-mario-bros/requirements.txt.bak
	cd gym-super-mario-bros; pip install -r requirements.txt && make deployment

.PHONY: make-nes-py
make-nes-py:
	# venv must be activated
	pip install setuptools
	cd nes-py && pip install -r requirements.txt && make
