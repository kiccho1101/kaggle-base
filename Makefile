build:
	docker-compose build

push:
	docker-compose push jupyter

pull:
	docker-compose pull jupyter

jupyter:
	docker-compose up jupyter 

jupyter-shell:
	docker-compose exec jupyter /bin/bash

run:
	docker-compose run --rm jupyter $(filter-out $@,$(MAKECMDGOALS))

shell:
	docker-compose run --rm jupyter /bin/bash

check:
	docker-compose run --rm jupyter flake8 /app/src