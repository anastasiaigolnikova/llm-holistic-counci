# 🏛️ LLM Holistic Council

A multi-model AI council that combines insights from multiple LLMs to provide comprehensive, well-reasoned answers. Inspired by [Andrej Karpathy's LLM Council](https://github.com/karpathy/llm-council).



## How It Works

1. **Stage 1: Individual Responses** - Multiple AI models (GPT-4, Claude, Perplexity, Llama) independently answer your question
2. **Stage 2: Peer Review** - Each model reviews the other models' answers
3. **Stage 3: Final Synthesis** - An independent judge model synthesizes all responses into a comprehensive answer

## Features

- 🤖 Multi-model council with GPT-4, Claude Sonnet 4, Perplexity Sonar Pro, and Llama 3.3
- 👨‍⚖️ Selectable independent judge for final synthesis
- 🌐 Automatic language detection (responds in the same language as your query)
- 📚 Archive to save and revisit past council sessions
- 🎨 Beautiful dark theme UI

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/llm-holistic-council.git
cd llm-holistic-council
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your API key

Create a `.env` file with your [OpenRouter](https://openrouter.ai/) API key:

```bash
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

### 4. Run the server

```bash
python server.py
```

Open http://localhost:8000 in your browser.

## Requirements

- Python 3.9+
- OpenRouter API key (get one at [openrouter.ai](https://openrouter.ai/))

## Tech Stack

- **Backend**: FastAPI + Python
- **Frontend**: Vanilla HTML/CSS/JS
- **API**: OpenRouter (unified access to multiple LLMs)

## Models Used

| Model | Provider | Role |
|-------|----------|------|
| GPT-4o | OpenAI | Council Member |
| Claude Sonnet 4 | Anthropic | Council Member / Judge |
| Sonar Reasoning Pro | Perplexity | Council Member |
| Llama 3.3 70B | Meta | Council Member |

## Credits

- Created by [Holistic Brand Lab](https://holisticbrandlab.com/)
- Inspired by [Andrej Karpathy's LLM Council](https://x.com/karpathy/status/1992381094667411768)

## License

MIT License - feel free to use and modify!

