# llux

llux is an AI chatbot for the [Matrix](https://matrix.org/) chat protocol. It uses local LLMs via [Ollama](https://ollama.ai/) and supports both image recognition and image generation. Each user in a Matrix room can set a unique personality (or system prompt), and conversations are kept per user, per channel. Model switching (OpenAI or Ollama) is also supported if you have multiple models configured.

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
   - **Copy `config.yaml-example` to `config.yaml`** (e.g., `cp config.yaml-example config.yaml`).
   - In your new `config.yaml`, fill in the relevant fields (Matrix server, username, password, channels, admin usernames, etc.). Also configure the Ollama section for your model settings and the Diffusers section for image generation (model, device, steps, etc.).

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
  - Include `modelname` to switch to that model.

- **.clear**  
  Reset llux for everyone, clearing all stored conversations and returning to the default settings.
