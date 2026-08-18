curl -sS -H 'x-uid: prod_user' \
  http://127.0.0.1:2024/alfred/agents |
  jq '{role, agents: [.agents[].id]}'