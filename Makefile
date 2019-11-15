build:
	docker-compose build

push:
	docker-compose push kaggle-base

pull:
	docker-compose pull kaggle-base

start-db:
	docker-compose build --no-cache postgres \
	&& docker-compose up -d postgres \
	&& sleep 15 \
	&& docker-compose up --build -d pgweb \
	&& docker-compose exec kaggle-base python db/initialize_db.py

reset-db:
	sudo rm -rf ./docker/postgres/data \
	&& docker-compose build --no-cache postgres \
	&& docker-compose up -d postgres \
	&& sleep 15 \
	&& docker-compose up --build -d pgweb \
	&& docker-compose exec kaggle-base python db/initialize_db.py

stop-db:
	docker-compose stop postgres pgweb

jupyter:
	docker-compose up -d kaggle-base \
	&& sleep 5 \
	&& docker-compose exec kaggle-base jupyter notebook list 

token:
	docker-compose exec kaggle-base jupyter notebook list 

run:
	docker-compose exec kaggle-base $(filter-out $@,$(MAKECMDGOALS))

shell:
	docker-compose exec kaggle-base /bin/bash

format:
	docker-compose exec kaggle-base isort /app/ --recursive \
	&& docker-compose exec kaggle-base black /app

check:
	docker-compose exec kaggle-base flake8 /app \
	&& docker-compose exec kaggle-base mypy /app

kfold: 
	docker-compose exec kaggle-base python k_fold/create.py $(filter-out $@,$(MAKECMDGOALS))

feature: 
	docker-compose exec kaggle-base python features/create.py $(filter-out $@,$(MAKECMDGOALS))

cv: 
	docker-compose exec kaggle-base python cross_validation/run.py $(filter-out $@,$(MAKECMDGOALS))

stats: 
	docker-compose exec kaggle-base python stats-db/create.py

train-and-predict: 
	docker-compose exec kaggle-base python train_and_predict/run.py $(filter-out $@,$(MAKECMDGOALS))

hp-tuning: 
	docker-compose exec kaggle-base python hp_tuning/run.py $(filter-out $@,$(MAKECMDGOALS))

stop-all:
	docker-compose down