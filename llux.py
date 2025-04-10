#!/usr/bin/env python3
"""
llux: A chatbot for Matrix with image recognition and image generation capabilities
Configure in config.yaml
Requires ollama, diffusers, matrix-nio
"""

import asyncio
import argparse
import datetime
import io
import json
import logging
import os
import random
import tempfile
import time
from openai import OpenAI
from typing import Optional, Dict, Any

import markdown
import yaml
from PIL import Image
from nio import AsyncClient, MatrixRoom, RoomMessageText, RoomMessageImage, UploadResponse
import torch
from diffusers import FluxPipeline
import ollama

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the llux chatbot.
    """
    parser = argparse.ArgumentParser(description="Matrix bot for Ollama chat")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()

class Llux:
    """
    llux chatbot class for Matrix rooms. Provides functionality for text-based conversation
    (via Ollama) and image generation (via a FluxPipeline), as well as basic administration
    commands and conversation management.
    """

    def __init__(self) -> None:
        """
        Initialize the llux chatbot, read config.yaml settings, prepare logging,
        initialize variables, load AI models, and create a temporary directory for files.
        """
        self.setup_logging()

        self.config_file = "config.yaml"
        self.logger.info(f"Loading config from {self.config_file}")

        with open(self.config_file, "r") as f:
            config = yaml.safe_load(f)

        # Matrix configuration
        matrix_config = config["matrix"]
        self.server = matrix_config["server"]
        self.username = matrix_config["username"]
        self.password = matrix_config["password"]
        self.channels = matrix_config["channels"]
        self.admins = matrix_config["admins"]
        
        # Diffusers configuration
        diffusers_config = config["diffusers"]
        self.diffusers_model = diffusers_config["model"]
        self.diffusers_device = diffusers_config["device"]
        self.diffusers_steps = diffusers_config["steps"]
        self.img_generation_confirmation = diffusers_config["img_generation_confirmation"]
        
        # Text-to-speech configuration
        # Example: reading from config["tts"] or you can just hard-code the base_url, model, etc.
        tts_config = config.get("tts", {})
        tts_url = tts_config.get("base_url")
        tts_api_key = tts_config.get("api_key", "not-needed")
        self.tts_model = tts_config.get("model", "kokoro")
        self.tts_voice = tts_config.get("voice", "af_sky+af_bella")  # single or multiple voicepacks
        
        # Initialize TTS client
        self.tts_client = OpenAI(
            base_url=tts_url,
            api_key=tts_api_key
        )
        
        self.awaiting_own_image = False  # Flag to listen for bot's own image
        self.awaiting_timeout = 0  # Timestamp to stop listening

        # Create Matrix client
        self.client = AsyncClient(self.server, self.username)
        self.join_time = datetime.datetime.now()
        self.messages: Dict[str, Dict[str, Any]] = {}
        self.temp_images: Dict[str, str] = {}

        # Ollama configuration
        ollama_config = config["ollama"]
        self.models = ollama_config["models"]
        self.default_model = ollama_config["default_model"]
        self.vision_model = ollama_config["vision_model"]
        self.model = self.models[self.default_model]

        # Conversation / model options
        options = ollama_config["options"]
        self.temperature = options["temperature"]
        self.top_p = options["top_p"]
        self.repeat_penalty = options["repeat_penalty"]
        self.defaults = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty
        }

        # Personality and prompt
        self.default_personality = ollama_config["personality"]
        self.personality = self.default_personality
        self.prompt = ollama_config["prompt"]

        # Verify requested models exist in config
        self.logger.info("Verifying requested models exist in config...")
        if self.default_model not in self.models:
            self.logger.warning(
                f"Default model '{self.default_model}' not found in config, "
                "using the first available model instead."
            )
            self.default_model = next(iter(self.models))
        if self.vision_model not in self.models:
            self.logger.warning(
                f"Vision model '{self.vision_model}' not found in config, "
                "using default model as fallback for vision tasks."
            )
            self.vision_model = self.default_model

        # Temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.logger.info(f"Created temporary directory at {self.temp_dir}")

        # Load Flux image generation pipeline
        self.logger.info(f"Loading {self.diffusers_model} pipeline for image generation...")
        try:
            self.pipe = FluxPipeline.from_pretrained(
                self.diffusers_model,
                torch_dtype=torch.bfloat16
            )
            self.pipe = self.pipe.to(self.diffusers_device)
            self.logger.info("Running a warm-up pass to minimize first-inference overhead...")
            start_time = time.time()
            _ = self.pipe(
                "warmup prompt",
                output_type="pil",
                num_inference_steps=1
            ).images[0]
            end_time = time.time()
            self.warmup_duration = end_time - start_time
            self.logger.info(
                f"Warmup complete. It took {self.warmup_duration:.2f} seconds for one step."
            )
        except Exception as e:
            self.logger.error(f"Error loading {self.diffusers_model} pipeline: {e}", exc_info=True)
            self.warmup_duration = None

    def setup_logging(self) -> None:
        """
        Set up file and console logging for the llux bot.
        """
        self.logger = logging.getLogger("llux")
        self.logger.setLevel(logging.DEBUG)

        # Log to file
        fh = logging.FileHandler("llux.log")
        fh.setLevel(logging.DEBUG)

        # Log to console
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    async def display_name(self, user: str) -> str:
        """
        Get the display name of a Matrix user, falling back to the user ID if there's an error.
        """
        try:
            name = await self.client.get_displayname(user)
            return name.displayname
        except Exception:
            return user

    async def send_message(self, channel: str, message: str) -> None:
        """
        Send a text message to the specified Matrix channel, rendering Markdown.
        """
        await self.client.room_send(
            room_id=channel,
            message_type="m.room.message",
            content={
                "msgtype": "m.text",
                "body": message,
                "format": "org.matrix.custom.html",
                "formatted_body": markdown.markdown(
                    message,
                    extensions=["fenced_code", "nl2br"]
                ),
            },
        )
        
    async def send_audio(self, channel: str, audio_path: str) -> None:
        """
        Upload and send an MP3 (or any audio) file to Matrix as m.audio.
        """
        try:
            filename = os.path.basename(audio_path)
            size_bytes = os.path.getsize(audio_path)
            with open(audio_path, "rb") as f:
                upload_response, upload_error = await self.client.upload(
                    f,
                    content_type="audio/mpeg",
                    filename=filename
                )
            if upload_error:
                self.logger.error(f"Failed to upload audio: {upload_error}")
                return
        
            self.logger.debug(f"Successfully uploaded audio, URI: {upload_response.content_uri}")
        
            await self.client.room_send(
                room_id=channel,
                message_type="m.room.message",
                content={
                    "msgtype": "m.audio",
                    "url": upload_response.content_uri,
                    "body": filename,
                    "info": {
                        "mimetype": "audio/mpeg",
                        "size": size_bytes,
                        # you can also add "duration" here in ms if you want
                    }
                }
            )
        except Exception as e:
            self.logger.error(f"Error sending audio: {e}", exc_info=True)
            await self.send_message(channel, f"Failed to send audio: {str(e)}")


    async def send_image(self, channel: str, image_path: str) -> str:
        """
        Send an image to a Matrix channel by uploading the file and then sending an m.image message.
        Returns the event ID of the sent message.
        """
        try:
            with open(image_path, "rb") as f:
                upload_response, upload_error = await self.client.upload(
                    f,
                    content_type="image/jpeg",
                    filename=os.path.basename(image_path)
                )
            if upload_error:
                self.logger.error(f"Failed to upload image: {upload_error}")
                return None
        
            self.logger.debug(f"Successfully uploaded image, URI: {upload_response.content_uri}")
        
            # Send the image message and capture the response
            send_response = await self.client.room_send(
                room_id=channel,
                message_type="m.room.message",
                content={
                    "msgtype": "m.image",
                    "url": upload_response.content_uri,
                    "body": os.path.basename(image_path),
                    "info": {
                        "mimetype": "image/jpeg",
                        "h": 300,
                        "w": 400,
                        "size": os.path.getsize(image_path)
                    }
                }
            )
        
            event_id = send_response.event_id
            self.logger.debug(f"Image sent with event ID: {event_id}")
            return event_id
        
        except Exception as e:
            self.logger.error(f"Error sending image: {e}", exc_info=True)
            await self.send_message(channel, f"Failed to send image: {str(e)}")
            return None

    async def download_image(self, mxc_url: str) -> Optional[str]:
        """
        Download an image from the given Matrix Content URI (mxc://...) and return a path to the file.
        Converts the image to RGB and saves as JPEG for consistency.
        """
        try:
            self.logger.debug(f"Downloading image from URL: {mxc_url}")
            response = await self.client.download(mxc_url)

            if response and response.body:
                self.logger.debug(f"Received image data, size: {len(response.body)} bytes")
                image = Image.open(io.BytesIO(response.body))
                self.logger.debug(f"Original image format: {image.format}, mode: {image.mode}")

                if image.mode != "RGB":
                    image = image.convert("RGB")
                    self.logger.debug("Converted image to RGB mode")

                fd, path = tempfile.mkstemp(suffix=".jpg", dir=self.temp_dir)
                image.save(path, "JPEG", quality=95)
                os.close(fd)
                self.logger.debug(f"Saved processed image to: {path}")

                return path

            self.logger.error("No image data in response")
            return None

        except Exception as e:
            self.logger.error(f"Error downloading/processing image: {e}", exc_info=True)
            return None
            
    async def generate_image(self, prompt: str) -> str:
        """
        Generate an image based on a text prompt using the FLUX pipeline.
    
        Args:
            prompt (str): The text description of the image to generate.
    
        Returns:
            str: The file path to the generated image.
        """
        self.logger.debug(f"Generating FLUX image for prompt: {prompt}")
        rand_seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator("cpu").manual_seed(rand_seed)
        
        try:
            result = self.pipe(
                prompt,
                output_type="pil",
                num_inference_steps=self.diffusers_steps,
                generator=generator
            )
            image = result.images[0]  # PIL Image
        
            fd, path = tempfile.mkstemp(suffix=".jpg", dir=self.temp_dir)
            os.close(fd)
            image = image.convert("RGB")
            image.save(path, "JPEG", quality=90)
        
            self.logger.debug(f"Generated image saved to {path}")
            return path
        
        except Exception as e:
            self.logger.error(f"Error generating image with FLUX: {e}", exc_info=True)
            raise e
    
    async def generate_tts(self, text: str) -> str:
        """
        Generate an audio file from text using a TTS service.
    
        Args:
            text (str): The text to convert to speech.
    
        Returns:
            str: The file path to the generated MP3 audio file.
        """
        self.logger.info(f"Generating TTS for text: '{text}'")
        
        fd, path = tempfile.mkstemp(suffix=".mp3", dir=self.temp_dir)
        os.close(fd)
        
        try:
            with self.tts_client.audio.speech.with_streaming_response.create(
                model=self.tts_model,
                voice=self.tts_voice,
                input=text,
                response_format="mp3"
            ) as response:
                response.stream_to_file(path)
        
            self.logger.debug(f"TTS audio saved to {path}")
            return path
        except Exception as e:
            self.logger.error(f"Error generating TTS audio: {e}", exc_info=True)
            raise e
    
    async def generate_and_send_tts(self, channel: str, text: str, user_sender: str) -> None:
        """
        Wrapper that calls generate_tts() and then uploads/sends the MP3 to Matrix.
        Also logs the event in conversation history.
        """
        try:
            # A small courtesy message to let user know TTS is being prepared
            confirmation_message = (
                f"{user_sender} Generating TTS for: '{text}'. Please wait..."
            )
            await self.send_message(channel, confirmation_message)
    
            audio_path = await self.generate_tts(text)
    
            # Upload and send the audio file
            await self.send_audio(channel, audio_path)
    
        except Exception as e:
            err_msg = f"Error generating TTS: {str(e)}"
            self.logger.error(err_msg, exc_info=True)
            await self.send_message(channel, err_msg)
        else:
            # Optionally store the TTS output in the conversation history
            await self.add_history(
                role="assistant",
                channel=channel,
                sender=user_sender,
                message=f"Generated TTS for text: {text}",
                image_path=None  # no image, but you could store an "audio_path" variant if you like
            )

    async def add_history(
        self,
        role: str,
        channel: str,
        sender: str,
        message: str,
        image_path: Optional[str] = None
    ) -> None:
        """
        Add a history entry for conversation context, storing it in self.messages.
        If the conversation grows beyond 24 items, prune from the front while
        preserving the system prompt.
        """
        if channel not in self.messages:
            self.messages[channel] = {}
        if sender not in self.messages[channel]:
            self.messages[channel][sender] = [
                {"role": "system", "content": f"{self.prompt[0]}{self.personality}{self.prompt[1]}"}
            ]

        history_entry = {"role": role, "content": message}
        if image_path:
            history_entry["images"] = [image_path]

        self.messages[channel][sender].append(history_entry)

        if len(self.messages[channel][sender]) > 24:
            # Keep the system prompt but prune old messages
            convo = self.messages[channel][sender]
            # If the first item is a system prompt, remove items after it until short enough
            if convo[0]["role"] == "system":
                self.messages[channel][sender] = [convo[0]] + convo[-23:]
            else:
                self.messages[channel][sender] = convo[-24:]

    async def respond(
        self,
        channel: str,
        sender: str,
        messages: Any,
        sender2: Optional[str] = None
    ) -> None:
        """
        Send conversation messages to Ollama for a response, handle tool calls if present,
        add the response to the history, and send it to the channel.
        """
        try:
            has_image = any("images" in msg for msg in messages)
            use_model_key = self.vision_model if has_image else self.default_model
    
            if use_model_key not in self.models:
                error_text = (
                    f"Requested model '{use_model_key}' not available. "
                    f"Available models: {', '.join(self.models.keys())}"
                )
                self.logger.error(error_text)
                await self.send_message(channel, error_text)
                return
    
            model_to_use = self.models[use_model_key]
    
            # Define available tools
            available_functions = {
                "generate_image": self.generate_image,
                "generate_tts": self.generate_tts,
            }
    
            log_messages = [
                {
                    **msg,
                    "images": [f"<image:{path}>" for path in msg.get("images", [])]
                } for msg in messages
            ]
            self.logger.debug(f"Sending to Ollama - model: {model_to_use}, messages: {json.dumps(log_messages)}")
    
            # Initial chat with tools
            response = ollama.chat(
                model=model_to_use,
                messages=messages,
                options={
                    "top_p": self.top_p,
                    "temperature": self.temperature,
                    "repeat_penalty": self.repeat_penalty,
                },
                tools=[self.generate_image, self.generate_tts]
            )
            response_message = response["message"]
    
            # Handle tool calls if present
            if response_message.get("tool_calls"):
                for tool in response_message["tool_calls"]:
                    function_name = tool["function"]["name"]
                    function_args = tool["function"]["arguments"]
                    function_to_call = available_functions.get(function_name)
    
                    if function_to_call:
                        self.logger.debug(f"Calling tool: {function_name} with args: {function_args}")
                        result = await function_to_call(**function_args)
    
                        if function_name == "generate_image":
                            event_id = await self.send_image(channel, result)
                            if event_id:
                                self.temp_images[event_id] = result
                                await self.add_history(
                                    "assistant", channel, sender, f"Generated image for: {function_args['prompt']}", result
                                )
                                # Set flag to listen for own image event
                                self.awaiting_own_image = True
                                self.awaiting_timeout = time.time() + 5
                                await asyncio.sleep(1)  # Allow event to arrive
                                self.awaiting_own_image = False
                        elif function_name == "generate_tts":
                            await self.send_audio(channel, result)
                            await self.add_history(
                                "assistant", channel, sender, f"Generated TTS for: {function_args['text']}"
                            )
    
                        # Add tool result to messages and get final response
                        messages.append({
                            "role": "tool",
                            "content": f"Tool {function_name} executed successfully",
                            "name": function_name
                        })
                        final_response = ollama.chat(
                            model=model_to_use,
                            messages=messages,
                            options={"top_p": self.top_p, "temperature": self.temperature, "repeat_penalty": self.repeat_penalty}
                        )
                        response_text = final_response["message"]["content"]
                    else:
                        response_text = f"Tool {function_name} not found"
            else:
                response_text = response_message["content"]
    
            await self.add_history("assistant", channel, sender, response_text)
            target_user = sender2 if sender2 else sender
            final_text = f"{target_user} {response_text.strip()}"
            await self.send_message(channel, final_text)
    
        except Exception as e:
            error_msg = f"Something went wrong: {e}"
            self.logger.error(error_msg, exc_info=True)
            await self.send_message(channel, error_msg)

    async def ai(
        self,
        channel: str,
        message: list[str],
        sender: str,
        event: Any,
        x: bool = False
    ) -> None:
        """
        Main logic for handling AI commands ('.ai', '@bot:', etc.) including replying to
        messages that contain images (vision model) and bridging conversation history
        for different users if needed ('.x').
        """
        try:
            relates_to = event.source["content"].get("m.relates_to", {})
            if "m.in_reply_to" in relates_to:
                reply_to_id = relates_to["m.in_reply_to"].get("event_id")
                if reply_to_id and reply_to_id in self.temp_images:
                    image_path = self.temp_images[reply_to_id]
                    await self.add_history("user", channel, sender, " ".join(message[1:]), image_path)
                else:
                    await self.add_history("user", channel, sender, " ".join(message[1:]))
            else:
                await self.add_history("user", channel, sender, " ".join(message[1:]))

            if x and len(message) > 2:
                name = message[1]
                original_message = " ".join(message[2:])
                name_id = name
                found = False

                if channel in self.messages:
                    for user_id in self.messages[channel]:
                        try:
                            username = await self.display_name(user_id)
                            if name.lower() == username.lower():
                                name_id = user_id
                                found = True
                                break
                        except Exception:
                            pass

                    if found:
                        if name_id in self.messages[channel]:
                            await self.add_history("user", channel, name_id, original_message)
                            await self.respond(channel, name_id, self.messages[channel][name_id], sender)
                        else:
                            await self.send_message(channel, f"No conversation history found for {name}")
                    else:
                        await self.send_message(channel, f"User {name} not found in this channel")
            else:
                await self.respond(channel, sender, self.messages[channel][sender])

        except Exception as e:
            self.logger.error(f"Error in ai(): {e}", exc_info=True)

    async def handle_image(self, room: MatrixRoom, event: RoomMessageImage) -> None:
        """
        Handle the arrival of an image message, downloading the image and storing
        a reference to it for possible AI conversation usage (in self.temp_images).
        """
        try:
            if event.url:
                image_path = await self.download_image(event.url)
                if image_path:
                    self.temp_images[event.event_id] = image_path

                    # Prune old images if we exceed 100 stored
                    if len(self.temp_images) > 100:
                        old_event_id = next(iter(self.temp_images))
                        old_path = self.temp_images.pop(old_event_id, "")
                        try:
                            os.remove(old_path)
                        except Exception:
                            self.logger.warning(f"Failed to remove old image {old_path}")
        except Exception as e:
            self.logger.error(f"Error handling image: {e}", exc_info=True)

    async def set_prompt(
        self,
        channel: str,
        sender: str,
        persona: Optional[str] = None,
        custom: Optional[str] = None,
        respond: bool = True
    ) -> None:
        """
        Set a new prompt for the conversation, either by applying a named persona or a custom
        string. Clear the conversation history for the user in this channel and optionally
        prompt the user to 'introduce' themselves.
        """
        try:
            self.messages[channel][sender].clear()
        except KeyError:
            pass

        if custom:
            prompt = custom
        elif persona:
            prompt = self.prompt[0] + persona + self.prompt[1]
        else:
            prompt = ""

        await self.add_history("system", channel, sender, prompt)

        if respond:
            await self.add_history("user", channel, sender, "introduce yourself")
            await self.respond(channel, sender, self.messages[channel][sender])

    async def reset(
        self,
        channel: str,
        sender: str,
        sender_display: str,
        stock: bool = False
    ) -> None:
        """
        Reset conversation history for a user in this channel. If stock is True, it applies
        'stock' settings without setting the default prompt; otherwise it sets the default
        prompt for the user.
        """
        if channel in self.messages:
            try:
                self.messages[channel][sender].clear()
            except KeyError:
                self.messages[channel] = {}
                self.messages[channel][sender] = []

        if not stock:
            await self.send_message(channel, f"{self.bot_id} reset to default for {sender_display}")
            await self.set_prompt(channel, sender, persona=self.personality, respond=False)
        else:
            await self.send_message(channel, f"Stock settings applied for {sender_display}")

    async def help_menu(self, channel: str, sender_display: str) -> None:
        """
        Present the help menu to the user from 'help.txt'. If the user is an admin,
        present admin commands as well.
        """
        with open("help.txt", "r") as f:
            help_content = f.read().split("~~~")
        help_menu = help_content[0]
        help_admin = help_content[1] if len(help_content) > 1 else ""

        await self.send_message(channel, help_menu)
        if sender_display in self.admins:
            await self.send_message(channel, help_admin)

    async def change_model(self, channel: str, model: Optional[str] = None) -> None:
        """
        Change the active Ollama model or reset to the default model if 'reset' is specified.
        If no model is given, display the current and available models.
        Added a check to see if the requested model is actually in config.
        """
        with open(self.config_file, "r") as f:
            config = yaml.safe_load(f)

        self.models = config["ollama"]["models"]
        if model:
            if model in self.models:
                self.model = self.models[model]
                await self.send_message(channel, f"Model set to **{self.model}**")
            elif model == "reset":
                self.model = self.models[self.default_model]
                await self.send_message(channel, f"Model reset to **{self.model}**")
            else:
                # Not found in config
                await self.send_message(
                    channel,
                    f"Requested model '{model}' not found in config. "
                    f"Available models: {', '.join(self.models.keys())}"
                )
        else:
            current_model = (
                f"**Current model**: {self.model}\n"
                f"**Available models**: {', '.join(sorted(self.models))}"
            )
            await self.send_message(channel, current_model)

    async def clear(self, channel: str) -> None:
        """
        Clear all conversation and image data for every user and reset the model and defaults.
        Also remove temporary image files from the filesystem.
        """
        self.messages.clear()
        self.model = self.models[self.default_model]
        self.personality = self.default_personality
        self.temperature, self.top_p, self.repeat_penalty = self.defaults.values()

        for path in list(self.temp_images.values()):
            try:
                os.remove(path)
                self.logger.debug(f"Removed temporary file: {path}")
            except Exception as e:
                self.logger.warning(f"Failed to remove temporary file {path}: {e}")
        self.temp_images.clear()

        await self.send_message(channel, "Bot has been reset for everyone")

    async def generate_and_send_image(
        self,
        channel: str,
        prompt: str,
        user_sender: str
    ) -> None:
        """
        Generate an image with the configured pipeline for the given prompt and send it to the Matrix room.
        Then add the generated image to the user's conversation history and wait for the event to be processed.
        """
        try:
            # Let user know we're working on it and estimate time based on warmup and steps
            estimated_time = None
            if self.warmup_duration is not None:
                estimated_time = self.warmup_duration * self.diffusers_steps
                confirmation_message = (
                    f"{user_sender} {self.img_generation_confirmation} '{prompt}'. "
                    f"Approximately {estimated_time:.2f} seconds to completion."
                )
            else:
                confirmation_message = (
                    f"{user_sender} {self.img_generation_confirmation} '{prompt}'. "
                    f"Time estimate unavailable."
                )
    
            await self.send_message(channel, confirmation_message)
    
            self.logger.info(f"User requested image for prompt: '{prompt}'")
            path = await self.generate_image(prompt)
    
            # Store the image path temporarily with a unique key (e.g., timestamp)
            temp_key = f"pending_{int(time.time()*1000)}"
            self.temp_images[temp_key] = path
    
            # Upload & send the image to the Matrix room, capturing the event ID
            event_id = await self.send_image(channel, path)
            if not event_id:
                raise Exception("Failed to send image, no event ID returned")
    
            # Store the generated image in conversation history
            await self.add_history(
                role="assistant",
                channel=channel,
                sender=user_sender,
                message=f"Generated image for prompt: {prompt}",
                image_path=path
            )
    
            # Set flag to listen for own image event for 5 seconds
            self.awaiting_own_image = True
            self.awaiting_timeout = time.time() + 5  # Wait up to 5 seconds
    
            # Wait briefly to allow the event to arrive
            await asyncio.sleep(1)
    
            # Reset the flag after timeout
            self.awaiting_own_image = False
    
        except Exception as e:
            err_msg = f"Error generating image: {str(e)}"
            self.logger.error(err_msg, exc_info=True)
            await self.send_message(channel, err_msg)

    async def handle_message(
        self,
        message: list[str],
        sender: str,
        sender_display: str,
        channel: str,
        event: Any
    ) -> None:
        """
        Primary message handler that routes user commands to the appropriate method.
        Commands recognized include:
          - .ai / @bot / .x        : conversation / AI responses
          - .persona / .custom     : set prompts
          - .reset / .stock        : reset conversation to default or stock
          - .help                  : display help menus
          - .img                   : generate and send an image
          - .tts                   : generate and send audio
          - .model / .clear        : admin commands
        """
        self.logger.debug(f"Handling message: {message[0]} from {sender_display}, event_id: {event.event_id}")

        user_commands = {
            ".ai": lambda: self.ai(channel, message, sender, event),
            f"{self.bot_id}:": lambda: self.ai(channel, message, sender, event),
            f"@{self.username}": lambda: self.ai(channel, message, sender, event),
            ".x": lambda: self.ai(channel, message, sender, event, x=True),
            ".persona": lambda: self.set_prompt(channel, sender, persona=" ".join(message[1:])),
            ".custom": lambda: self.set_prompt(channel, sender, custom=" ".join(message[1:])),
            ".reset": lambda: self.reset(channel, sender, sender_display),
            ".stock": lambda: self.reset(channel, sender, sender_display, stock=True),
            ".help": lambda: self.help_menu(channel, sender_display),
            ".img": lambda: self.generate_and_send_image(channel, " ".join(message[1:]), sender),
            ".tts": lambda: self.generate_and_send_tts(channel, " ".join(message[1:]), sender),
        }

        admin_commands = {
            ".model": lambda: self.change_model(channel, model=message[1] if len(message) > 1 else None),
            ".clear": lambda: self.clear(channel),
        }

        command = message[0]

        if command in user_commands:
            await user_commands[command]()
        elif sender_display in self.admins and command in admin_commands:
            await admin_commands[command]()
        else:
            self.logger.debug(f"Unknown command or unauthorized: {command}")

    async def message_callback(self, room: MatrixRoom, event: Any) -> None:
        """
        Callback to handle messages (text or image) that arrive in the Matrix room.
        Process own images when awaiting_own_image is True.
        """
        message_time = datetime.datetime.fromtimestamp(event.server_timestamp / 1000)
        
        # Check if message is recent and either from another user or the bot when awaiting an image
        if message_time > self.join_time and (
            event.sender != self.username or (self.awaiting_own_image and time.time() < self.awaiting_timeout)
        ):
            try:
                if isinstance(event, RoomMessageImage):
                    await self.handle_image(room, event)
                    # Reset the flag after processing an image if it was the bot's own
                    if event.sender == self.username:
                        self.awaiting_own_image = False
                elif isinstance(event, RoomMessageText):
                    message = event.body.split(" ")
                    sender = event.sender
                    sender_display = await self.display_name(sender)
                    channel = room.room_id
                    await self.handle_message(message, sender, sender_display, channel, event)
            except Exception as e:
                self.logger.error(f"Error in message_callback: {e}", exc_info=True)

    def cleanup(self) -> None:
        """
        Remove all temporary image files and attempt to delete the temporary directory.
        Called at the end of execution or upon shutdown to clean up resources.
        """
        self.logger.info("Cleaning up temporary files")
        for path in self.temp_images.values():
            try:
                os.remove(path)
                self.logger.debug(f"Removed temporary file: {path}")
            except Exception as e:
                self.logger.warning(f"Failed to remove temporary file {path}: {e}")
        try:
            os.rmdir(self.temp_dir)
            self.logger.debug(f"Removed temporary directory: {self.temp_dir}")
        except Exception as e:
            self.logger.warning(f"Failed to remove temporary directory {self.temp_dir}: {e}")

    async def main(self) -> None:
        """
        Main entry point for the llux bot. Logs in, joins configured channels, and starts
        a long-running sync loop. Upon exit, runs cleanup.
        """
        try:
            login_response = await self.client.login(self.password)
            self.logger.info(f"Login response: {login_response}")

            self.bot_id = await self.display_name(self.username)
            self.logger.info(f"Bot display name: {self.bot_id}")

            for channel in self.channels:
                try:
                    await self.client.join(channel)
                    self.logger.info(f"{self.bot_id} joined {channel}")
                except Exception as e:
                    self.logger.error(f"Couldn't join {channel}: {e}")

            self.client.add_event_callback(self.message_callback, (RoomMessageText, RoomMessageImage))

            self.logger.info("Starting sync loop")
            await self.client.sync_forever(timeout=30000, full_state=True)
        except Exception as e:
            self.logger.error(f"Error in main: {e}", exc_info=True)
        finally:
            self.cleanup()

if __name__ == "__main__":
    args = parse_args()
    bot = Llux()
    if args.debug:
        bot.logger.setLevel(logging.DEBUG)
        for handler in bot.logger.handlers:
            handler.setLevel(logging.DEBUG)
    asyncio.run(bot.main())
