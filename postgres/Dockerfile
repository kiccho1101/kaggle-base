FROM postgres:10.5

ARG PROJECT_NAME

ENV PROJECT_NAME "$PROJECT_NAME"

RUN echo "${PROJECT_NAME}"

RUN echo "CREATE DATABASE ${PROJECT_NAME};" >> /docker-entrypoint-initdb.d/init.sql

EXPOSE 5432