build:
	docker-compose build

push:
	docker-compose push jupyter

pull:
	docker-compose pull jupyter

start-db:
	docker-compose up --build -d postgres pgweb

stop-db:
	docker-compose stop postgres pgweb

jupyter:
	docker-compose up jupyter 

run:
	docker-compose exec jupyter $(filter-out $@,$(MAKECMDGOALS))

shell:
	docker-compose exec jupyter /bin/bash

format:
	docker-compose exec jupyter isort /app/src --recursive 
	docker-compose exec jupyter black /app/src

check:
	docker-compose exec jupyter flake8 /app/src
	docker-compose exec jupyter mypy /app/src

stop-all:
	docker-compose down