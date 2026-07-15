📣 sta_agent_engine — what's new in create_chat_model

A few things landed recently that change how you configure models — most of them mean less code on your side.

1️⃣ Any provider, zero code — no settings class needed
You no longer create/register a settings class per provider. Pass any name and the env prefix is derived automatically as {NAME}_*:

create_chat_model("acme")   # reads ACME_API_KEY, ACME_BASE_URL, ACME_MODEL, ...

dotenv
ACME_API_KEY=sk-...
ACME_BASE_URL=https://api.acme.internal/v1
ACME_MODEL=some-large-model

Built-ins still work (llmaas, mistral, custom, openai, …). custom uses no prefix (bare API_KEY/BASE_URL/MODEL), and openai reads the canonical OPENAI_* vars. A ready-to-fill starter lives at .env.provider.example.

2️⃣ Capacity tiers, straight from .env
One provider can now carry several model slots, picked with tier=:

dotenv
ACME_MODEL=default-model
ACME_BIG_MODEL=big-model
ACME_SMALL_MODEL=cheap-fast-model
ACME_THINKING_MODEL=reasoning-model

create_chat_model("acme", tier="small")     # falls back to ACME_MODEL if unset
create_chat_model("acme", tier="thinking")  # cascade: thinking → big → default

3️⃣ Vision models — multimodal=True
Set {NAME}_MULTIMODAL_MODEL and ask for it as a capability: create_chat_model("acme", multimodal=True). It never silently hands you a text-only model (raises instead of dropping your images).

4️⃣ Reasoning/thinking control — reasoning_effort= (new)
One vocabulary, translated per model family into whatever the model actually honors (Mistral reasoning_effort, Nemotron/Qwen chat_template_kwargs):

create_chat_model("acme", model="nemotron-3-super-120b", reasoning_effort="low")
create_chat_model("mistral", reasoning_effort="off")   # → "none" on the wire

Omit it → nothing changes. Unsupported values warn, never break. Your explicit extra_body/model_kwargs always win. Gateway alias hiding the model name? Pin with reasoning_family=. New thinking model? One register_reasoning_family(...) call at startup.

5️⃣ Also handy: per-call BYOK — create_chat_model("acme", provider_api_key="sk-...", provider_base_url="https://...") injects credentials at call time without touching env.

⚠️ Heads-up — two DeprecationWarnings that will become errors:
- Relying on a provider's implicit default model (no model=, no {NAME}_MODEL) → raises in 0.11.0. Always set the env var or pass model=.
- Passing the model positionally (create_chat_model("gpt-4o") — that's read as a provider!) → raises in 0.10.0. Use create_chat_model(provider=..., model=...).

📖 Docs:
- create_chat_model guide (providers, env contract, tiers, multimodal): 
- Reasoning guide (wire tables, custom families, limitations): 

Questions / odd gateway behavior → ping me.