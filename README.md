# Multi-Provider Chatbot (Chainlit)

A Chainlit chat app that lets you talk to Claude, GPT, or Gemini, plus run Kling image/video generation. Pick a provider, choose a model, set temperature, and chat in the browser.

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
5. (Optional) Enable sidebar auth:
   ```env
   CHAINLIT_AUTH_USERNAME=admin
   CHAINLIT_AUTH_PASSWORD=your_password
   ```

## Run

```bash
chainlit run main.py -w
```

This starts a browser UI. If you prefer no auto-reload, drop `-w`.

## Notes

- Chain commands: `/new` resets chat, `/temp 0.7` sets temperature, `/model your-model` picks a model, `/provider Claude|GPT|Gemini|Kling` switches providers, `/media` shows generated media history.
- Kling Image → Video and Multi‑Image → Video are launched by choosing the Kling provider. The flow guides you through prompts, images (upload, URL, or Base64), and options. Results return a video URL and are indexed in `/media`.
- Provider/model pickers appear as action buttons; if a model isn't shown, set it with `/model ...`.
- Thread titles are generated from the first real prompt (GPT‑5.1) and won’t rename on provider/model selection.
- Errors from providers are shown inline.
- To point at Azure/OpenAI-compatible gateways, set `OPENAI_BASE_URL` (defaults to `https://api.openai.com/v1`).
