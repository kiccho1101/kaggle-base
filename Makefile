build:
	docker-compose build

push:
	docker-compose push jupyter

pull:
	docker-compose pull jupyter

start-db:
	&& docker-compose up --build -d postgres \
	&& sleep 15 \
	&& docker-compose up --build -d pgweb \
	&& docker-compose exec jupyter python db/init.py

reset-db:
	sudo rm -rf postgres/data \
	&& docker-compose build --no-cache postgres \
	&& docker-compose up -d postgres \
	&& sleep 15 \
	&& docker-compose up --build -d pgweb \
	&& docker-compose exec jupyter python db/init.py

stop-db:
	docker-compose stop postgres pgweb

jupyter:
	docker-compose up -d jupyter 

token:
	docker-compose exec jupyter jupyter notebook list 

run:
	docker-compose exec jupyter $(filter-out $@,$(MAKECMDGOALS))

shell:
	docker-compose exec jupyter /bin/bash

format:
	docker-compose exec jupyter isort /app/ --recursive 
	docker-compose exec jupyter black /app/

check:
	docker-compose exec jupyter flake8 /app
	docker-compose exec jupyter mypy /app

stop-all:
	docker-compose down