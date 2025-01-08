# llux

llux is an AI chatbot for the [Matrix](https://matrix.org/) chat protocol. It uses local LLMs via [Ollama]([https://ollama.ai/](https://ollama.com) for chat and image recognition, and offers image generation via [FLUX.1](https://github.com/black-forest-labs/flux). Each user in a Matrix room can set a unique personality (or system prompt), and conversations are kept per user, per channel. Model switching is also supported if you have multiple models installed and configured.

## Getting Started

1. **Install Ollama**  
   You’ll need Ollama to run local LLMs. A quick install:  
   `curl https://ollama.ai/install.sh | sh`

   Then pull your preferred model(s) with `ollama pull <modelname>`.

2. **Install matrix-nio**  
   `pip3 install matrix-nio`

3. **Set Up Your Bot**

   - Create a Matrix account for your bot (on a server of your choice).
   - Record the server, username, and password.
   - Add these details, along with any custom models, to your `config.json`.

4. **Run llux**  
   `python3 llux.py`

## Usage

- **.ai message** or **botname: message**  
  Basic conversation or roleplay prompt.

- **.x username message**  
  Interact with another user’s chat history (use the display name of that user).

- **.persona personality**  
  Set or change to a specific roleplaying personality.

- **.custom prompt**  
  Override the default personality with a custom system prompt.

- **.reset**  
  Clear your personal conversation history and revert to the preset personality.

- **.stock**  
  Clear your personal conversation history, but do not apply any system prompt.

### Admin Commands

- **.model modelname**

  - Omit `modelname` to show the current model and available options.
  - Include `modelname` to switch.

- **.clear**  
  Reset llux for everyone, clearing all stored conversations and returning to the default settings.
