
.PHONY: all
all:
	@echo 'Specify target'
	@echo 'You must activate venv before running `make setup`'

.PHONY: setup
setup:	setup-gym-smb

.PHONY: setup-gym-smb
setup-gym-smb:	make-nes-py
	sed -i.bak "s%^nes-py.*%$(shell pwd)/nes-py/dist/nes_py-8.2.1.tar.gz%" gym-super-mario-bros/requirements.txt
	cd gym-super-mario-bros; pip install -r requirements.txt && make
	pip install --no-index --find-links=$(shell pwd)/gym-super-mario-bros/dist/gym_super_mario_bros-8.0.0-py3-none-any.whl gym-super-mario-bros

.PHONY: make-nes-py
make-nes-py:
	# venv must be activated
	cd nes-py && pip install -r requirements.txt && make
