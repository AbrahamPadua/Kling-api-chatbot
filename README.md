# Multi-Provider Chatbot (Chainlit)

A Chainlit chat app that lets you talk to Claude, GPT, or Gemini. Pick a provider, choose a model, set temperature, and chat in the browser.

## Setup

1. Install Python 3.9+.
2. (Recommended) Create and activate a virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Add your API keys in `.env` (already created):
   ```env
   ANTHROPIC_API_KEY=your_anthropic_key_here
   OPENAI_API_KEY=your_openai_key_here
   GEMINI_API_KEY=your_gemini_key_here
   KLING_ACCESS_KEY=your_kling_access_key_here
   KLING_SECRET_KEY=your_kling_secret_key_here
   ```

## Run

```bash
chainlit run main.py -w
```

This starts a browser UI. If you prefer no auto-reload, drop `-w`.

## Notes

- Chain commands: `/new` resets chat, `/temp 0.7` sets temperature, `/model your-model` picks a model, `/provider Claude|GPT|Gemini` switches providers.
- Kling Multi-Image → Video: run `/kling` to launch the guided flow for 1–4 images (URL or Base64) plus prompt/negative prompt. The result is returned as a video URL.
- Provider/model pickers appear as action buttons; if a model isn't shown, set it with `/model ...`.
- Histories are kept per provider during the session.
- Errors from providers are shown inline.
- To point at Azure/OpenAI-compatible gateways, set `OPENAI_BASE_URL` (defaults to `https://api.openai.com/v1`).
