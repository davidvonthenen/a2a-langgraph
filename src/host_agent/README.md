# An example host agent frontend

This host agent now uses [LangGraph](https://github.com/langchain-ai/langgraph)
to coordinate conversations between the weather and Airbnb specialists. The
graph enforces a simple travel policy:

* Whenever a user asks about Airbnb lodging, the host first requests a forecast
  from the weather agent.
* Hazardous conditions automatically block downstream lodging suggestions and
  the user is told why the plan was paused.

This makes the flow reproducible and easy to audit because every policy step is
captured as a deterministic graph edge.

## Running the example

1. Create a `.env` file using the `example.env` file as a template.

2. Start the demo:

   ```bash
   uv run .
   ```

   Then open the Gradio UI and try a prompt such as:

   > Plan a spring Airbnb stay in Denver for four friends.

   The host will fetch a weather outlook first and only provide rental options
   if the forecast looks safe.
