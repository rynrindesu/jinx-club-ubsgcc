# kotlin-spring-boot-template

These instruction are to help you solve a test challenge "Calculate Square".

## Calculate Square

### Instructions

This challenge requires you to return the squared of two numbers given an itput.

### Endpoint
Create an endpoint `/square` that accepts a JSON payload over `POST` described below.

### Input

Example:
`{ "input": 5 }`

### Output

Example:
`25`

## Explanation

This project is built for [spring boot](https://spring.io/projects/spring-boot) and is meant to run a REST server which we will be using mainly for UBS Coding Challenge.

Make sure you have Java environment setup on your local machine. This template uses java-17 so do download it.

Running `Boot.kt` will start up a rest controller which you can hit `localhost:8080/ping` to get back the response `pong`.

Currently, we have set up a `SquareController.kt` which exposes an endpoint `/square` for `POST` requests to be made with the given input, and would return the squared of the input as output.

To extend this template further, add more endpoints in the `controllers` directory and expose more endpoints according to the given questions.

Note, due to the limitation of render in running JVM applications, we have to run them in docker containers. 

This template has already created a `Dockerfile` for render to dockerize the JVM applications. 

All you need to do is run `./gradlew clean build` and commit the jar files to your git repository which would be picked up by render for deployment. 

If after pushing to gitlab, and render is able to build the docker container successfully yet you are unable to `/ping` the base endpoint, it is highly possible that your port is not exposed. To do so, create an environment variable on render `PORT` and give it the value of `8080` which is the port exposed by Dockerfile.