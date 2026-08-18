{
  "agents": [
    {
      "enabled": true,
      "id": "ext_customer_research",
      "name": "Customer Research Agent",
      "specialty": "Customer-specific business research",
      "description": "Queries the customer's approved knowledge and returns an evidence-backed business analysis.",
      "url": "https://customer-agent.example.com",
      "graph_id": "customer_research",
      "api_key_env": "ALFRED_CUSTOMER_RESEARCH_API_KEY",
      "use_for": [
        "Questions that require the customer's private business knowledge",
        "Cross-checking Alfred's conclusions against the customer's specialist"
      ],
      "examples": [
        "Analyze the operational impact of this proposal using the customer's internal knowledge."
      ]
    }
