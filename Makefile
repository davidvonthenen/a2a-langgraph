.PHONY: airbnb_agent weather_agent host_agent test

airbnb_agent:
	python -m src.airbnb_agent

weather_agent:
	python -m src.weather_agent

host_agent:
	python -m src.host_agent

test:
	python -m pytest
