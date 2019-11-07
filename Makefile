build:
	docker-compose build

push:
	docker-compose push jupyter

pull:
	docker-compose pull jupyter

jupyter:
	docker-compose up jupyter 

run:
	docker-compose exec jupyter $(filter-out $@,$(MAKECMDGOALS))

shell:
	docker-compose exec jupyter /bin/bash

check:
	docker-compose exec jupyter flake8 /app/src