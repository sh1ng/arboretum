build:
	mkdir -p build && \
	cd build && \
	cmake .. && \
	$(MAKE) -j

build/libarboretum.so: build

wheel: build/libarboretum.so
	cp build/libarboretum.so python-package/arboretum/
	rm -rf python-package/dist && \
	cd python-package && \
	python3 setup.py sdist bdist_wheel --plat-name=manylinux1_x86_64

clean:
	rm -rf build
	rm -rf python-package/dist

.PHONY: docker_wheel
docker_wheel:
	docker build -t cuda . && \
	docker run -v "/home/sh1ng/dev/arboretum:/repo" -d --name arboretum-wheel -it cuda && \
	docker exec arboretum-wheel bash -c 'cd repo; make clean; make wheel' && \
	docker cp arboretum-wheel:/repo/python-package/dist/ . && \
	docker stop arboretum-wheel
	docker rm arboretum-wheel