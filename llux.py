#!/usr/bin/env python3
"""
llux: A chatbot for Matrix with image recognition and image generation capabilities
Configure in config.yaml
Requires ollama, diffusers, matrix-nio, requests
"""

import asyncio
import argparse
import datetime
import io
import json
import logging
import os
import random
import re
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
    parser = argparse.ArgumentParser(description="Matrix bot for Ollama chat")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()

class Llux:
    def __init__(self) -> None:
        self.setup_logging()
        self.config_file = "config.yaml"
        self.logger.info(f"Loading config from {self.config_file}")

        with open(self.config_file, "r") as f:
            config = yaml.safe_load(f)

        matrix_config = config["matrix"]
        self.server = matrix_config["server"]
        self.username = matrix_config["username"]
        self.password = matrix_config["password"]
        self.channels = matrix_config["channels"]
        self.admins = matrix_config["admins"]

        diffusers_config = config["diffusers"]
        self.diffusers_model = diffusers_config["model"]
        self.diffusers_device = diffusers_config["device"]
        self.diffusers_steps = diffusers_config["steps"]
        self.img_generation_confirmation = diffusers_config["img_generation_confirmation"]

        tts_config = config.get("tts", {})
        tts_url = tts_config.get("base_url")
        tts_api_key = tts_config.get("api_key", "not-needed")
        self.tts_model = tts_config.get("model", "kokoro")
        self.tts_voice = tts_config.get("voice", "af_sky+af_bella")

        self.tts_client = OpenAI(base_url=tts_url, api_key=tts_api_key)

        self.awaiting_own_image = False
        self.awaiting_timeout = 0

        self.client = AsyncClient(self.server, self.username)
        self.join_time = datetime.datetime.now()
        self.messages: Dict[str, Dict[str, Any]] = {}
        self.temp_images: Dict[str, str] = {}

        ollama_config = config["ollama"]
        self.models = ollama_config["models"]
        self.default_model = ollama_config["default_model"]
        self.vision_model = ollama_config["vision_model"]
        self.model = self.models[self.default_model]

        options = ollama_config["options"]
        self.temperature = options["temperature"]
        self.top_p = options["top_p"]
        self.repeat_penalty = options["repeat_penalty"]
        self.defaults = {"temperature": self.temperature, "top_p": self.top_p, "repeat_penalty": self.repeat_penalty}

        self.default_personality = ollama_config["personality"]
        self.personality = self.default_personality
        self.prompt = ollama_config["prompt"]

        if self.default_model not in self.models:
            self.logger.warning(f"Default model '{self.default_model}' not found, using first available.")
            self.default_model = next(iter(self.models))
        if self.vision_model not in self.models:
            self.logger.warning(f"Vision model '{self.vision_model}' not found, using default.")
            self.vision_model = self.default_model

        self.temp_dir = tempfile.mkdtemp()
        self.logger.info(f"Created temporary directory at {self.temp_dir}")

        self.logger.info(f"Loading {self.diffusers_model} pipeline...")
        try:
            self.pipe = FluxPipeline.from_pretrained(self.diffusers_model, torch_dtype=torch.bfloat16)
            self.pipe = self.pipe.to(self.diffusers_device)
            self.logger.info("Running warm-up pass...")
            start_time = time.time()
            _ = self.pipe("warmup prompt", output_type="pil", num_inference_steps=1).images[0]
            self.warmup_duration = time.time() - start_time
            self.logger.info(f"Warmup complete. Took {self.warmup_duration:.2f} seconds.")
        except Exception as e:
            self.logger.error(f"Error loading pipeline: {e}", exc_info=True)
            self.warmup_duration = None

    def setup_logging(self) -> None:
        self.logger = logging.getLogger("llux")
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler("llux.log")
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    async def display_name(self, user: str) -> str:
        try:
            name = await self.client.get_displayname(user)
            return name.displayname
        except Exception:
            return user

    async def send_message(self, channel: str, message: str) -> None:
        await self.client.room_send(
            room_id=channel,
            message_type="m.room.message",
            content={
                "msgtype": "m.text",
                "body": message,
                "format": "org.matrix.custom.html",
                "formatted_body": markdown.markdown(message, extensions=["fenced_code", "nl2br"])
            }
        )

    async def send_audio(self, channel: str, audio_path: str) -> None:
        try:
            filename = os.path.basename(audio_path)
            size_bytes = os.path.getsize(audio_path)
            with open(audio_path, "rb") as f:
                upload_response, upload_error = await self.client.upload(f, content_type="audio/mpeg", filename=filename)
            if upload_error:
                self.logger.error(f"Failed to upload audio: {upload_error}")
                return
            self.logger.debug(f"Uploaded audio, URI: {upload_response.content_uri}")
            await self.client.room_send(
                room_id=channel,
                message_type="m.room.message",
                content={
                    "msgtype": "m.audio",
                    "url": upload_response.content_uri,
                    "body": filename,
                    "info": {"mimetype": "audio/mpeg", "size": size_bytes}
                }
            )
        except Exception as e:
            self.logger.error(f"Error sending audio: {e}", exc_info=True)
            await self.send_message(channel, f"Failed to send audio: {str(e)}")

    async def send_image(self, channel: str, image_path: str) -> str:
        try:
            with open(image_path, "rb") as f:
                upload_response, upload_error = await self.client.upload(f, content_type="image/jpeg", filename=os.path.basename(image_path))
            if upload_error:
                self.logger.error(f"Failed to upload image: {upload_error}")
                return None
            self.logger.debug(f"Uploaded image, URI: {upload_response.content_uri}")
            send_response = await self.client.room_send(
                room_id=channel,
                message_type="m.room.message",
                content={
                    "msgtype": "m.image",
                    "url": upload_response.content_uri,
                    "body": os.path.basename(image_path),
                    "info": {"mimetype": "image/jpeg", "h": 300, "w": 400, "size": os.path.getsize(image_path)}
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
        try:
            self.logger.debug(f"Downloading image from URL: {mxc_url}")
            response = await self.client.download(mxc_url)
            if response and response.body:
                self.logger.debug(f"Received image data, size: {len(response.body)} bytes")
                image = Image.open(io.BytesIO(response.body))
                if image.mode != "RGB":
                    image = image.convert("RGB")
                fd, path = tempfile.mkstemp(suffix=".jpg", dir=self.temp_dir)
                image.save(path, "JPEG", quality=95)
                os.close(fd)
                self.logger.debug(f"Saved image to: {path}")
                return path
            self.logger.error("No image data in response")
            return None
        except Exception as e:
            self.logger.error(f"Error downloading image: {e}", exc_info=True)
            return None

    async def generate_image(self, prompt: str) -> str:
        self.logger.debug(f"Generating FLUX image for prompt: {prompt}")
        rand_seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator("cpu").manual_seed(rand_seed)
        try:
            result = self.pipe(prompt, output_type="pil", num_inference_steps=self.diffusers_steps, generator=generator)
            image = result.images[0]
            fd, path = tempfile.mkstemp(suffix=".jpg", dir=self.temp_dir)
            os.close(fd)
            image.convert("RGB").save(path, "JPEG", quality=90)
            self.logger.debug(f"Generated image saved to {path}")
            return path
        except Exception as e:
            self.logger.error(f"Error generating image: {e}", exc_info=True)
            raise e

    async def generate_tts(self, text: str) -> str:
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
            self.logger.error(f"Error generating TTS: {e}", exc_info=True)
            raise e

    async def describe_image(self, image_path: str) -> str:
        self.logger.debug(f"Describing image at: {image_path}")
        try:
            response = ollama.chat(
                model=self.models[self.vision_model],
                messages=[{"role": "user", "content": "Describe this image", "images": [image_path]}],
                options={"temperature": self.temperature, "top_p": self.top_p, "repeat_penalty": self.repeat_penalty}
            )
            return response["message"]["content"]
        except Exception as e:
            self.logger.error(f"Error describing image: {e}", exc_info=True)
            return f"Error describing image: {str(e)}"

    async def transcribe_audio(self, audio_path: str) -> str:
        self.logger.debug(f"Transcribing audio at: {audio_path}")
        try:
            with open(audio_path, "rb") as audio_file:
                response = self.tts_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            return response
        except Exception as e:
            self.logger.error(f"Error transcribing audio: {e}", exc_info=True)
            return f"Error transcribing audio: {str(e)}"

    async def generate_and_send_tts(self, channel: str, text: str, user_sender: str) -> None:
        try:
            confirmation_message = f"{user_sender} Generating TTS for: '{text}'. Please wait..."
            await self.send_message(channel, confirmation_message)
            audio_path = await self.generate_tts(text)
            await self.send_audio(channel, audio_path)
            await self.add_history("assistant", channel, user_sender, f"Generated TTS for text: {text}")
        except Exception as e:
            err_msg = f"Error generating TTS: {str(e)}"
            self.logger.error(err_msg, exc_info=True)
            await self.send_message(channel, err_msg)

    async def add_history(self, role: str, channel: str, sender: str, message: str, image_path: Optional[str] = None) -> None:
        if channel not in self.messages:
            self.messages[channel] = {}
        if sender not in self.messages[channel]:
            self.messages[channel][sender] = [{"role": "system", "content": f"{self.prompt[0]}{self.personality}{self.prompt[1]}"}]
        history_entry = {"role": role, "content": message}
        if image_path:
            history_entry["images"] = [image_path]
        self.messages[channel][sender].append(history_entry)
        if len(self.messages[channel][sender]) > 24:
            convo = self.messages[channel][sender]
            if convo[0]["role"] == "system":
                self.messages[channel][sender] = [convo[0]] + convo[-23:]
            else:
                self.messages[channel][sender] = convo[-24:]

    async def respond(self, channel: str, sender: str, messages: Any, sender2: Optional[str] = None) -> None:
        try:
            use_model_key = self.default_model
            if use_model_key not in self.models:
                error_text = f"Requested model '{use_model_key}' not available. Available: {', '.join(self.models.keys())}"
                self.logger.error(error_text)
                await self.send_message(channel, error_text)
                return

            model_to_use = self.models[use_model_key]
            available_functions = {
                "generate_image": self.generate_image,
                "generate_tts": self.generate_tts,
                "describe_image": self.describe_image,
                "transcribe_audio": self.transcribe_audio,
            }
            log_messages = [
                {**msg, "images": [f"<image:{path}>" for path in msg.get("images", [])] if "images" in msg else []}
                for msg in messages
            ]
            self.logger.debug(f"Sending to Ollama - model: {model_to_use}, messages: {json.dumps(log_messages)}")
            response = ollama.chat(
                model=model_to_use,
                messages=messages,
                options={"top_p": self.top_p, "temperature": self.temperature, "repeat_penalty": self.repeat_penalty},
                tools=[self.generate_image, self.generate_tts, self.describe_image, self.transcribe_audio]
            )
            response_message = response["message"]

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
                                await self.add_history("assistant", channel, sender, f"Generated image for: {function_args['prompt']}", result)
                                self.awaiting_own_image = True
                                self.awaiting_timeout = time.time() + 5
                                await asyncio.sleep(1)
                                self.awaiting_own_image = False
                            return
                        elif function_name == "generate_tts":
                            await self.send_audio(channel, result)
                            await self.add_history("assistant", channel, sender, f"Generated TTS for: {function_args['text']}")
                            return
                        elif function_name in ["describe_image", "transcribe_audio"]:
                            await self.add_history("assistant", channel, sender, result)
                            target_user = sender2 if sender2 else sender
                            final_text = f"{target_user} {result.strip()}"
                            await self.send_message(channel, final_text)
                            return
                        messages.append({"role": "tool", "content": result, "name": function_name})
                        final_response = ollama.chat(
                            model=model_to_use,
                            messages=messages,
                            options={"top_p": self.top_p, "temperature": self.temperature, "repeat_penalty": self.repeat_penalty}
                        )
                        response_text = re.sub(r'<think>.*?</think>', '', final_response["message"]["content"], flags=re.DOTALL).strip()
                    else:
                        response_text = f"Tool {function_name} not found"
            else:
                response_text = re.sub(r'<think>.*?</think>', '', response_message["content"], flags=re.DOTALL).strip()

            await self.add_history("assistant", channel, sender, response_text)
            target_user = sender2 if sender2 else sender
            final_text = f"{target_user} {response_text}"
            await self.send_message(channel, final_text)

        except Exception as e:
            error_msg = f"Something went wrong: {e}"
            self.logger.error(error_msg, exc_info=True)
            await self.send_message(channel, error_msg)

    async def ai(self, channel: str, message: list[str], sender: str, event: Any, x: bool = False) -> None:
        try:
            full_message = " ".join(message[1:])
            relates_to = event.source["content"].get("m.relates_to", {})
            if "m.in_reply_to" in relates_to:
                reply_to_id = relates_to["m.in_reply_to"].get("event_id")
                if reply_to_id and reply_to_id in self.temp_images:
                    image_path = self.temp_images[reply_to_id]
                    await self.add_history("user", channel, sender, full_message, image_path)
                else:
                    await self.add_history("user", channel, sender, full_message)
            else:
                lines = event.body.split("\n")
                if lines and lines[0].startswith("> "):
                    quoted_text = lines[0][2:].strip()
                    event_id_match = re.search(r'\$[A-Za-z0-9_-]+', quoted_text)
                    if event_id_match and event_id_match.group(0) in self.temp_images:
                        await self.add_history("user", channel, sender, "\n".join(lines[1:]), self.temp_images[event_id_match.group(0)])
                    else:
                        await self.add_history("user", channel, sender, full_message)
                else:
                    await self.add_history("user", channel, sender, full_message)

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
        try:
            if event.url:
                image_path = await self.download_image(event.url)
                if image_path:
                    self.temp_images[event.event_id] = image_path
                    if len(self.temp_images) > 100:
                        old_event_id = next(iter(self.temp_images))
                        old_path = self.temp_images.pop(old_event_id, "")
                        try:
                            os.remove(old_path)
                        except Exception:
                            self.logger.warning(f"Failed to remove old image {old_path}")
        except Exception as e:
            self.logger.error(f"Error handling image: {e}", exc_info=True)

    async def set_prompt(self, channel: str, sender: str, persona: Optional[str] = None, custom: Optional[str] = None, respond: bool = True) -> None:
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

    async def reset(self, channel: str, sender: str, sender_display: str, stock: bool = False) -> None:
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
        with open("help.txt", "r") as f:
            help_content = f.read().split("~~~")
        help_menu = help_content[0]
        help_admin = help_content[1] if len(help_content) > 1 else ""
        await self.send_message(channel, help_menu)
        if sender_display in self.admins:
            await self.send_message(channel, help_admin)

    async def change_model(self, channel: str, model: Optional[str] = None) -> None:
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
                await self.send_message(channel, f"Model '{model}' not found. Available: {', '.join(self.models.keys())}")
        else:
            current_model = f"**Current model**: {self.model}\n**Available models**: {', '.join(sorted(self.models))}"
            await self.send_message(channel, current_model)

    async def clear(self, channel: str) -> None:
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

    async def generate_and_send_image(self, channel: str, prompt: str, user_sender: str) -> None:
        try:
            estimated_time = self.warmup_duration * self.diffusers_steps if self.warmup_duration else None
            confirmation_message = (
                f"{user_sender} {self.img_generation_confirmation} '{prompt}'. "
                f"Approximately {estimated_time:.2f} seconds to completion." if estimated_time
                else f"{user_sender} {self.img_generation_confirmation} '{prompt}'. Time estimate unavailable."
            )
            await self.send_message(channel, confirmation_message)
            self.logger.info(f"User requested image for prompt: '{prompt}'")
            path = await self.generate_image(prompt)
            temp_key = f"pending_{int(time.time()*1000)}"
            self.temp_images[temp_key] = path
            event_id = await self.send_image(channel, path)
            if not event_id:
                raise Exception("Failed to send image, no event ID returned")
            await self.add_history("assistant", channel, user_sender, f"Generated image for prompt: {prompt}", path)
            self.awaiting_own_image = True
            self.awaiting_timeout = time.time() + 5
            await asyncio.sleep(1)
            self.awaiting_own_image = False
        except Exception as e:
            err_msg = f"Error generating image: {str(e)}"
            self.logger.error(err_msg, exc_info=True)
            await self.send_message(channel, err_msg)

    async def handle_message(self, message: list[str], sender: str, sender_display: str, channel: str, event: Any) -> None:
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
        message_time = datetime.datetime.fromtimestamp(event.server_timestamp / 1000)
        if message_time > self.join_time and (
            event.sender != self.username or (self.awaiting_own_image and time.time() < self.awaiting_timeout)
        ):
            try:
                if isinstance(event, RoomMessageImage):
                    await self.handle_image(room, event)
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
