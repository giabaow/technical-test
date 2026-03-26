COMPOSE = docker-compose

.PHONY: build run-pipeline run deploy stop logs logs-deploy test clean curl-health curl-predict curl-languages mlflow

# docker
build:       
	$(COMPOSE) build

mlflow:
	docker compose up -d mlflow

run-pipeline:  
	$(COMPOSE) up -d mlflow
	$(COMPOSE) up --abort-on-container-exit data-prep train evaluate

run:      
	$(COMPOSE) up -d mlflow
	$(COMPOSE) up --abort-on-container-exit data-prep train evaluate
	$(COMPOSE) up -d deploy

deploy:      
	$(COMPOSE) up -d mlflow
	$(COMPOSE) up deploy

stop:          
	$(COMPOSE) down

logs:       
	$(COMPOSE) logs -f

logs-deploy:  
	$(COMPOSE) logs -f deploy


# test
test:          ## Run pytest test suite (requires local Python env)
	pip install -q pytest httpx fastapi scikit-learn joblib pandas gradio uvicorn
	pytest tests/ -v

# clear
clean:         ## Remove containers, images, volumes, and cached data
	$(COMPOSE) down --rmi all --volumes --remove-orphans
	rm -rf data/processed data/model data/results


# api test

curl-health: #Test the /health endpoint
	curl -s http://localhost:8080/health | python3 -m json.tool

curl-predict: #Test the /predict endpoint with a sample sentence
	curl -s -X POST http://localhost:8080/predict \
	  -H "Content-Type: application/json" \
	  -d '{"text": "Bonjour, comment allez-vous?"}' | python3 -m json.tool

curl-languages: #List all supported languages
	curl -s http://localhost:8080/languages | python3 -m json.tool
