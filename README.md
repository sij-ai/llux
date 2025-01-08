<img src="https://sij.ai/sij/llux/raw/branch/main/welcome.jpg" width="400" alt="llux welcome image">

# llux

llux is an AI chatbot for the [Matrix](https://matrix.org/) chat protocol. It uses local LLMs via [Ollama](https://ollama.ai/) for chat and image recognition, and offers image generation via [Diffusers](https://github.com/huggingface/diffusers), specifically [FLUX.1](https://github.com/black-forest-labs/flux). Each user in a Matrix room can set a unique personality (or system prompt), and conversations are kept per user, per channel. Model switching is also supported if you have multiple models installed and configured.

You're welcome to try the bot out on [We2.ee](https://we2.ee/about) at [#ai:we2.ee](https://we2.ee/@@ai).

## Getting Started

1. **Install Ollama**  
   You’ll need Ollama to run local LLMs (text and multimodal). A quick install:

   ```bash
   curl https://ollama.ai/install.sh | sh
   ```

   Then pull your preferred model(s) with:

   ```bash
   ollama pull <modelname>
   ```

   > **Note**: For image recognition in multimodal chat, you’ll need a vision model. This can be—but doesn’t have to be—the same as your primary chat model.

2. **Create a Python Environment (Recommended)**  
   You can use either `conda/mamba` or `venv`:

   ```bash
   # Using conda/mamba:
   mamba create -n llux python=3.10
   conda activate llux

   # or using Python's built-in venv:
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**  
   Install all required Python libraries from `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

   This will install:

   - `matrix-nio` for Matrix connectivity
   - `diffusers` for image generation
   - `ollama` for local LLMs
   - `torch` for the underlying deep learning framework
   - `pillow` for image manipulation
   - `markdown`, `pyyaml`, etc.

4. **Set Up Your Bot**

   - Create a Matrix account for your bot (on a server of your choice).
   - Record the server, username, and password.
   - **Copy `config.yaml-example` to `config.yaml`** (e.g., `cp config.yaml-example config.yaml`).
   - In your new `config.yaml`, fill in the relevant fields (Matrix server, username, password, channels, admin usernames, etc.). Also configure the Ollama section for your model settings and the Diffusers section for image generation (model, device, steps, etc.).

5. **Run llux**
   ```bash
   python3 llux.py
   ```
   If you’re using a virtual environment, ensure it’s activated first.

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
