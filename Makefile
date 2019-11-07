build:
	docker-compose build

push:
	docker-compose push kaggle-base

pull:
	docker-compose pull kaggle-base

jupyter:
	docker-compose up kaggle-base 

jupyter-shell:
	docker-compose exec kaggle-base /bin/bash

run:
	docker-compose run --rm kaggle-base $(filter-out $@,$(MAKECMDGOALS))

shell:
	docker-compose run --rm kaggle-base /bin/bash

check:
	docker-compose run --rm kaggle-base flake8 /app/src