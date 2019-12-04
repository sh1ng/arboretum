.PHONY: wheel
wheel:
	rm -rf python-package/dist && \
	rm -rf build && \
	mkdir build && \
	cd build && \
	cmake .. && \
	$(MAKE) -j && \
	cd ../python-package && \
	python3 setup.py sdist bdist_wheel --plat-name=linux_x86_64

.PHONY: docker_wheel
docker_wheel:
	docker build -t cuda . && \
	docker run -v "/home/sh1ng/dev/arboretum:/repo" -d --name arboretum-wheel -it cuda && \
	docker exec arboretum-wheel bash -c 'cd repo; make wheel' && \
	docker cp arboretum-wheel:/repo/python-package/dist/ . && \
	docker stop arboretum-wheel