build:
	docker-compose build

push:
	docker-compose push jupyter

pull:
	docker-compose pull jupyter

start-db:
	docker-compose build --no-cache postgres \
	&& docker-compose up -d postgres \
	&& sleep 15 \
	&& docker-compose up --build -d pgweb \
	&& docker-compose up --build -d metabase \
	&& sleep 5 \
	&& docker-compose restart metabase \
	&& docker-compose exec jupyter python db/init.py

reset-db:
	sudo rm -rf ./docker/postgres/data \
	&& sudo rm -rf ./docker/metabase/data \
	&& docker-compose build --no-cache postgres \
	&& docker-compose up -d postgres \
	&& sleep 15 \
	&& docker-compose up --build -d pgweb \
	&& docker-compose up --build -d metabase \
	&& sleep 5 \
	&& docker-compose restart metabase \
	&& docker-compose exec jupyter python db/init.py

stop-db:
	docker-compose stop postgres pgweb metabase

jupyter:
	docker-compose up -d jupyter \
	&& sleep 5 \
	&& docker-compose exec jupyter jupyter notebook list 

token:
	docker-compose exec jupyter jupyter notebook list 

run:
	docker-compose exec jupyter $(filter-out $@,$(MAKECMDGOALS))

shell:
	docker-compose exec jupyter /bin/bash

format:
	docker-compose exec jupyter isort /app/ --recursive \
	&& docker-compose exec jupyter black /app

check:
	docker-compose exec jupyter flake8 /app \
	&& docker-compose exec jupyter mypy /app

stop-all:
	docker-compose down