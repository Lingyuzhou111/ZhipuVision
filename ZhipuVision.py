import os
import json
import base64
import requests
import plugins
import time
import copy
import jwt
from bridge.context import Context, ContextType
from bridge.reply import Reply, ReplyType
from channel.chat_message import ChatMessage
from common.log import logger
from config import conf
from plugins import Plugin, Event, EventContext, EventAction
from PIL import Image
from io import BytesIO
import urllib.request
import re
import cv2
import numpy as np
import tempfile

@plugins.register(name="ZhipuVision", desc="智谱AI视觉模型插件", version="1.6", author="Lingyuzhou")
class ZhipuVision(Plugin):
    def __init__(self):
        super().__init__()
        curdir = os.path.dirname(__file__)
        config_path = os.path.join(curdir, "config.json")
        
        default_config = {
            "api": {
                "base_url": "https://open.bigmodel.cn/api/paas/v4",
                "model": "glm-4v-flash",
                "timeout": 180,
                "key": "",
                "temperature": 0.8,
                "top_p": 0.7
            },
            "image": {"max_size": 10, "max_pixels": 4096},
            "video": {"max_size": 20, "max_duration": 60},
            "trigger_keywords": {
                "image_analysis": ["z识图", "智谱识图"],
                "reverse_keyword": ["z反推"],
                "i2v_keyword": ["z运镜"],
                "vfx_keyword": ["z特效"],
                "storyboard_keyword": ["z分镜"],
                "video_analysis": ["z视频", "智谱识视频"]
            },
            "prompts": {
                "image_prompt": "请描述这张图片。",
                "reverse_prompt": "请分析这张图片，并尝试给出一些可以生成类似图片的提示词。",
                "video_prompt": "请描述这个视频的主要内容。",
                "ask_for_image": "请发送图片。",
                "ask_for_video_url": "请提供媒体文件的URL链接。",
                "image_process_fail": "图片处理失败，请检查图片格式或稍后再试。",
                "video_process_fail": "视频处理失败，请检查视频URL或稍后再试。",
                "api_error": "抱歉，AI服务暂时出现问题，请稍后再试。",
                "timeout_reply": "操作超时，请重新发送指令。",
                "cancel_reply": "操作已取消。",
                "flash_model_no_upload_support": "当前模型 (glm-4v-flash) 不支持处理上传的图片，请使用图片的公开URL链接或取消操作后使用URL发送指令。",
                "flash_model_no_local_support": "当前模型 (glm-4v-flash) 不支持处理引用的本地图片，请使用图片的公开URL链接。"
            },
            "wait_timeout": 180,
            "temp_dir": "tmp"
        }

        try:
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    loaded_config = json.load(f)
                _merged_config = default_config.copy()
                for key, value in loaded_config.items():
                    if isinstance(value, dict) and isinstance(_merged_config.get(key), dict):
                        current_default_dict = _merged_config[key]
                        new_dict_from_config = value
                        for k, v_config in new_dict_from_config.items():
                            current_default_dict[k] = v_config 
                    else:
                        _merged_config[key] = value
                self.config = _merged_config
            else:
                logger.warning(f"[ZhipuVision] config.json not found at {config_path}. Using default config and attempting to save.")
                self.config = default_config
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(self.config, f, indent=4, ensure_ascii=False)

            self.api_key = self.config.get("api", {}).get("key", "")
            if not self.api_key:
                raise Exception("[ZhipuVision] API key not found in config.json or is empty.")

            self.keywords = self.config.get("trigger_keywords", default_config["trigger_keywords"])
            self.prompts = self.config.get("prompts", default_config["prompts"])
            
            # 读取提示词文件
            self.i2v_prompt = self._load_prompt_file(curdir, "i2v_prompt.txt")
            self.vfx_prompt = self._load_prompt_file(curdir, "vfx_prompt.txt")
            self.storyboard_prompt = self._load_prompt_file(curdir, "storyboard_prompt.txt")
            
            self.temp_dir = os.path.join(curdir, self.config.get("temp_dir", "tmp"))
            os.makedirs(self.temp_dir, exist_ok=True)

            logger.info(f"[ZhipuVision] Plugin initialized successfully with model: {self.config.get('api',{}).get('model')}")
            
            self.handlers[Event.ON_HANDLE_CONTEXT] = self.on_handle_context
            self.waiting_for_file = {}
            self.current_context = None
            
        except Exception as e:
            logger.error(f"[ZhipuVision] Failed to initialize plugin: {e}")
            raise

    def _load_prompt_file(self, curdir: str, filename: str) -> str:
        """读取提示词文件内容"""
        try:
            file_path = os.path.join(curdir, filename)
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                logger.info(f"[ZhipuVision] Successfully loaded prompt file: {filename}")
                return content
            else:
                logger.warning(f"[ZhipuVision] Prompt file not found: {filename}")
                return ""
        except Exception as e:
            logger.error(f"[ZhipuVision] Error loading prompt file {filename}: {e}")
            return ""

    def _truncate_if_long(self, text: str, max_length: int = 50) -> str:
        if isinstance(text, str) and len(text) > max_length:
            return text[:max_length] + "..."
        return text

    def _get_image_data(self, msg: ChatMessage, image_path: str) -> bytes | None:
        """Reliably gets image bytes from various sources."""
        try:
            if isinstance(image_path, bytes):
                return image_path

            if os.path.isfile(image_path):
                with open(image_path, 'rb') as f:
                    return f.read()
            
            if msg and hasattr(msg, 'content'):
                if isinstance(msg.content, bytes):
                    return msg.content
                elif isinstance(msg.content, str) and os.path.isfile(msg.content):
                    with open(msg.content, 'rb') as f:
                        return f.read()

            if image_path.startswith('http://') or image_path.startswith('https://'):
                return self._download_media(image_path)

            if image_path.startswith('tmp/') and not os.path.isabs(image_path):
                potential_path = os.path.join(self.temp_dir, os.path.basename(image_path))
                if os.path.isfile(potential_path):
                    with open(potential_path, 'rb') as f:
                        return f.read()
            
            if msg and hasattr(msg, '_prepare_fn') and hasattr(msg, '_prepared') and not msg._prepared:
                logger.debug("[ZhipuVision] Trying to prepare/download image via msg._prepare_fn()")
                msg._prepare_fn()
                msg._prepared = True
                time.sleep(1)
                if hasattr(msg, 'content'):
                    if isinstance(msg.content, bytes):
                        return msg.content
                    elif isinstance(msg.content, str) and os.path.isfile(msg.content):
                        with open(msg.content, 'rb') as f:
                            return f.read()
            
            logger.error(f"[ZhipuVision] Could not get image data from path: {image_path}")
            return None
        except Exception as e:
            logger.error(f"[ZhipuVision] Error in _get_image_data for {image_path}: {e}")
            return None

    def _process_image(self, image_content_input, msg: ChatMessage = None) -> str:
        """Processes image_content_input and returns a PURE base64 string."""
        try:
            image_data_bytes = self._get_image_data(msg, image_content_input)
            if not image_data_bytes:
                raise Exception("Failed to get image data.")

            size_mb = len(image_data_bytes) / (1024 * 1024)
            if size_mb > self.config.get("image",{}).get("max_size", 10):
                raise Exception(f"图片大小超过限制 ({size_mb:.1f}MB > {self.config.get('image',{}).get('max_size', 10)}MB)")
            
            img = Image.open(BytesIO(image_data_bytes))
            output_buffer = BytesIO()
            if img.mode == 'RGBA' or img.mode == 'LA' or (img.mode == 'P' and 'transparency' in img.info):
                img = img.convert('RGB')
            img.save(output_buffer, format="JPEG")
            image_data_bytes_jpeg = output_buffer.getvalue()

            if max(img.size) > self.config.get("image",{}).get("max_pixels", 4096):
                raise Exception(f"图片尺寸超过限制 ({max(img.size)} > {self.config.get('image',{}).get('max_pixels', 4096)})")
            
            # Return PURE base64 string
            return base64.b64encode(image_data_bytes_jpeg).decode('utf-8')
            
        except Exception as e:
            logger.error(f"[ZhipuVision] Image processing error: {e}")
            raise

    def _extract_url_from_text(self, text: str) -> str | None:
        if not text: return None
        url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
        urls = re.findall(url_pattern, text)
        return urls[0] if urls else None

    def _handle_image_upload_while_waiting(self, e_context: EventContext, user_id: str, image_content_input):
        wait_info = self.waiting_for_file.pop(user_id, None)
        if not wait_info:
            return

        context = e_context['context']
        reply = Reply()
        
        action_type_key = wait_info.get("type")
        prompt_config_mapping = {"image": "image_prompt", "reverse": "reverse_prompt", "i2v": "i2v_prompt", "vfx": "vfx_prompt", "storyboard": "storyboard_prompt"}
        prompt_text_key = prompt_config_mapping.get(action_type_key, "image_prompt")
        
        # 根据功能类型选择提示词
        if action_type_key == "i2v":
            prompt_text_for_api = wait_info.get("question", self.i2v_prompt) if self.i2v_prompt else "请分析这张图片并生成图生视频的运镜提示词。"
        elif action_type_key == "vfx":
            prompt_text_for_api = wait_info.get("question", self.vfx_prompt) if self.vfx_prompt else "请分析这张图片并生成视觉特效提示词。"
        elif action_type_key == "storyboard":
            prompt_text_for_api = wait_info.get("question", self.storyboard_prompt) if self.storyboard_prompt else "请分析这张图片并生成分镜脚本。"
        else:
            prompt_text_for_api = wait_info.get("question", self.prompts.get(prompt_text_key, "请描述这张图片。"))

        current_model = self.config.get("api", {}).get("model")
        if current_model == "glm-4v-flash":
            logger.info(f"[ZhipuVision] Model {current_model} does not support uploaded images. Replying to user.")
            reply = Reply(ReplyType.TEXT, self.prompts.get("flash_model_no_upload_support", "当前模型 (glm-4v-flash) 不支持处理上传的图片，请使用图片的公开URL链接或取消操作后使用URL发送指令。"))
            e_context['reply'] = reply
            e_context.action = EventAction.BREAK_PASS
            return
        
        # For other models, proceed with Base64 processing
        try:
            processed_pure_base64_data = self._process_image(image_content_input, msg=context.kwargs.get('msg'))
            
            api_messages_payload = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": processed_pure_base64_data}},
                        {"type": "text", "text": prompt_text_for_api}
                    ]
                }
            ]
            response = self._call_glm_api(api_messages_payload)
            
            if response and "choices" in response and response["choices"]:
                reply_text = response["choices"][0]["message"]["content"]
            else:
                specific_api_message = None
                if isinstance(response, dict):
                    error_info = response.get("error")
                    if isinstance(error_info, dict):
                        specific_api_message = error_info.get("message")
                
                if specific_api_message:
                    if not isinstance(specific_api_message, str):
                        specific_api_message = str(specific_api_message)
                    reply_text = specific_api_message
                else:
                    reply_text = self.prompts.get("api_error", "抱歉，处理失败")
            reply = Reply(ReplyType.TEXT, reply_text)
        except Exception as e:
            logger.error(f"[ZhipuVision] Error processing waiting image for {user_id} ({action_type_key}): {e}")
            reply = Reply(ReplyType.TEXT, self.prompts.get("image_process_fail", f"处理失败: {str(e)}"))
        
        e_context['reply'] = reply
        e_context.action = EventAction.BREAK_PASS

    def on_handle_context(self, e_context: EventContext):
        context = e_context['context']
        self.current_context = context 
        msg_from_context: ChatMessage = context 
        actual_msg_object = msg_from_context.kwargs.get('msg')
        user_id = msg_from_context.get('from_user_id') or "unknown_user"
        reply = Reply()

        if actual_msg_object and \
           hasattr(actual_msg_object, 'is_processed_image_quote') and \
           actual_msg_object.is_processed_image_quote and \
           hasattr(actual_msg_object, 'referenced_image_path') and \
           actual_msg_object.referenced_image_path and \
           context.type == ContextType.TEXT:
            
            cmd_text = context.content.strip()
            matched_keyword_type = None
            action_prompt_key = None
            keywords_to_strip_from_cmd = []
            
            for kw_type, kws in self.keywords.items():
                if kw_type in ["image_analysis", "reverse_keyword", "i2v_keyword", "vfx_keyword", "storyboard_keyword"]:
                    for kw_val in kws:
                        if cmd_text.startswith(kw_val):
                            matched_keyword_type = kw_type
                            if kw_type == "image_analysis":
                                action_prompt_key = "image_prompt"
                            elif kw_type == "reverse_keyword":
                                action_prompt_key = "reverse_prompt"
                            elif kw_type == "i2v_keyword":
                                action_prompt_key = "i2v_prompt"
                            elif kw_type == "vfx_keyword":
                                action_prompt_key = "vfx_prompt"
                            elif kw_type == "storyboard_keyword":
                                action_prompt_key = "storyboard_prompt"
                            keywords_to_strip_from_cmd = [kw_val]
                            break
                if matched_keyword_type: break
            
            if matched_keyword_type:
                user_specific_prompt_text = cmd_text
                if keywords_to_strip_from_cmd:
                    user_specific_prompt_text = cmd_text[len(keywords_to_strip_from_cmd[0]):].strip()
                
                # 根据功能类型选择提示词
                if action_prompt_key == "i2v_prompt":
                    final_prompt_for_api = user_specific_prompt_text or (self.i2v_prompt if self.i2v_prompt else "请分析这张图片并生成图生视频的运镜提示词。")
                elif action_prompt_key == "vfx_prompt":
                    final_prompt_for_api = user_specific_prompt_text or (self.vfx_prompt if self.vfx_prompt else "请分析这张图片并生成视觉特效提示词。")
                elif action_prompt_key == "storyboard_prompt":
                    final_prompt_for_api = user_specific_prompt_text or (self.storyboard_prompt if self.storyboard_prompt else "请分析这张图片并生成分镜脚本。")
                else:
                    final_prompt_for_api = user_specific_prompt_text or self.prompts.get(action_prompt_key, "请描述这张图片。")
                
                current_model_for_ref = self.config.get("api", {}).get("model")
                if current_model_for_ref == "glm-4v-flash":
                    logger.info(f"[ZhipuVision] Model {current_model_for_ref} does not support referenced (local) images. Replying to user.")
                    reply = Reply(ReplyType.INFO, self.prompts.get("flash_model_no_local_support", "当前模型 (glm-4v-flash) 不支持处理引用的本地图片，请使用图片的公开URL链接。"))
                    e_context['reply'] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                # For other models, proceed with Base64 processing for referenced image
                try:
                    logger.info(f"[ZhipuVision] Processing referenced image for {action_prompt_key} by {user_id}. Path: {actual_msg_object.referenced_image_path}")
                    processed_pure_base64_data = self._process_image(actual_msg_object.referenced_image_path, msg=actual_msg_object)
                    
                    api_messages_payload = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": processed_pure_base64_data}},
                                {"type": "text", "text": final_prompt_for_api}
                            ]
                        }
                    ]
                    api_response = self._call_glm_api(api_messages_payload)
                    if api_response and "choices" in api_response and api_response["choices"]:
                        reply_text = api_response["choices"][0]["message"]["content"]
                    else:
                        specific_api_message = None
                        if isinstance(api_response, dict):
                            error_info = api_response.get("error")
                            if isinstance(error_info, dict):
                                specific_api_message = error_info.get("message")
                        
                        if specific_api_message:
                            if not isinstance(specific_api_message, str):
                                specific_api_message = str(specific_api_message)
                            reply_text = specific_api_message
                        else:
                            reply_text = self.prompts.get("api_error", "抱歉，处理失败")
                    reply = Reply(ReplyType.TEXT, reply_text)
                except Exception as e:
                    logger.error(f"[ZhipuVision] Error processing referenced image: {e}")
                    reply = Reply(ReplyType.TEXT, self.prompts.get("image_process_fail", f"处理失败: {str(e)}"))
                
                e_context['reply'] = reply
                e_context.action = EventAction.BREAK_PASS
                return

        current_time = time.time()
        for uid, wait_info_check in list(self.waiting_for_file.items()):
            if current_time - wait_info_check.get("time", 0) > self.config.get("wait_timeout", 180):
                logger.info(f"[ZhipuVision] Clearing expired waiting state for user {uid}, type {wait_info_check.get('type')}")
                self.waiting_for_file.pop(uid, None)
                if uid == user_id:
                    reply = Reply(ReplyType.INFO, self.prompts.get("timeout_reply", "操作超时，请重新发送指令。"))
                    e_context['reply'] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return

        if user_id in self.waiting_for_file:
            if context.type == ContextType.IMAGE:
                logger.info(f"[ZhipuVision] User {user_id} uploaded image while in waiting state: {self.waiting_for_file[user_id]['type']}")
                self._handle_image_upload_while_waiting(e_context, user_id, context.content)
                return
            elif context.type == ContextType.TEXT and any(cancel_kw in context.content.strip().lower() for cancel_kw in ["z取消", "z结束"]):
                logger.info(f"[ZhipuVision] User {user_id} cancelled waiting state: {self.waiting_for_file[user_id]['type']}")
                self.waiting_for_file.pop(user_id, None)
                reply = Reply(ReplyType.INFO, self.prompts.get("cancel_reply", "操作已取消。"))
                e_context['reply'] = reply
                e_context.action = EventAction.BREAK_PASS
                return

        if context.type == ContextType.TEXT:
            text_content = context.content.strip()
            
            matched_keyword_str = None
            command_type_key = None
            prompt_config_key = None

            for type_key, kws_list in self.keywords.items():
                for kw_candidate in kws_list:
                    if text_content.startswith(kw_candidate):
                        matched_keyword_str = kw_candidate
                        command_type_key = type_key
                        if type_key == "image_analysis":
                            prompt_config_key = "image_prompt"
                        elif type_key == "reverse_keyword":
                            prompt_config_key = "reverse_prompt"
                        elif type_key == "i2v_keyword":
                            prompt_config_key = "i2v_prompt"
                        elif type_key == "vfx_keyword":
                            prompt_config_key = "vfx_prompt"
                        elif type_key == "storyboard_keyword":
                            prompt_config_key = "storyboard_prompt"
                        elif type_key == "video_analysis":
                            prompt_config_key = "video_prompt"
                        break
                if matched_keyword_str: break
            
            if matched_keyword_str and command_type_key and prompt_config_key:
                text_after_keyword = text_content[len(matched_keyword_str):].strip()
                extracted_url = self._extract_url_from_text(text_after_keyword)
                user_specific_question = text_after_keyword
                if extracted_url:
                    user_specific_question = user_specific_question.replace(extracted_url, "").strip()
                
                # 根据功能类型选择提示词
                if prompt_config_key == "i2v_prompt":
                    final_prompt_for_api = user_specific_question or (self.i2v_prompt if self.i2v_prompt else "请分析这张图片并生成图生视频的运镜提示词。")
                elif prompt_config_key == "vfx_prompt":
                    final_prompt_for_api = user_specific_question or (self.vfx_prompt if self.vfx_prompt else "请分析这张图片并生成视觉特效提示词。")
                elif prompt_config_key == "storyboard_prompt":
                    final_prompt_for_api = user_specific_question or (self.storyboard_prompt if self.storyboard_prompt else "请分析这张图片并生成包含连贯动作变化和明确运镜的分镜脚本，至少包含2个以上的镜头切换。")
                else:
                    final_prompt_for_api = user_specific_question or self.prompts.get(prompt_config_key, "请描述此内容。")

                if command_type_key in ["image_analysis", "reverse_keyword", "i2v_keyword", "vfx_keyword", "storyboard_keyword"]:
                    if extracted_url:
                        logger.info(f"[ZhipuVision] Processing direct URL for {command_type_key} by {user_id}: {extracted_url}")
                        try:
                            # For direct URL, image_url object first, then text object
                            api_messages_payload = [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "image_url", "image_url": {"url": extracted_url}},
                                        {"type": "text", "text": final_prompt_for_api}
                                    ]
                                }
                            ]
                            api_response = self._call_glm_api(api_messages_payload)
                            if api_response and "choices" in api_response and api_response["choices"]:
                                reply_text = api_response["choices"][0]["message"]["content"]
                            else:
                                specific_api_message = None
                                if isinstance(api_response, dict):
                                    error_info = api_response.get("error")
                                    if isinstance(error_info, dict):
                                        specific_api_message = error_info.get("message")
                                
                                if specific_api_message:
                                    if not isinstance(specific_api_message, str):
                                        specific_api_message = str(specific_api_message)
                                    reply_text = specific_api_message
                                else:
                                    reply_text = self.prompts.get("api_error", "抱歉，处理失败")
                            reply = Reply(ReplyType.TEXT, reply_text)
                        except Exception as e:
                            logger.error(f"[ZhipuVision] Error processing direct URL for {command_type_key}: {e}")
                            reply = Reply(ReplyType.TEXT, self.prompts.get("image_process_fail", f"处理失败: {str(e)}"))
                        e_context['reply'] = reply
                        e_context.action = EventAction.BREAK_PASS
                        return
                    else:
                        # 根据功能类型设置等待类型
                        if command_type_key == "image_analysis":
                            wait_type_for_file = "image"
                        elif command_type_key == "reverse_keyword":
                            wait_type_for_file = "reverse"
                        elif command_type_key == "i2v_keyword":
                            wait_type_for_file = "i2v"
                        elif command_type_key == "vfx_keyword":
                            wait_type_for_file = "vfx"
                        elif command_type_key == "storyboard_keyword":
                            wait_type_for_file = "storyboard"
                        else:
                            wait_type_for_file = "image"
                        
                        self.waiting_for_file[user_id] = {"type": wait_type_for_file, "time": time.time(), "question": final_prompt_for_api}
                        reply_content = self.prompts.get("ask_for_image", "请发送图片。") 
                        if user_specific_question :
                            reply_content += f"\n已记录您的问题：'{user_specific_question}'"
                        reply = Reply(ReplyType.INFO, reply_content)
                        e_context['reply'] = reply
                        e_context.action = EventAction.BREAK_PASS
                        return
                
                elif command_type_key == "video_analysis":
                    if not extracted_url:
                        reply = Reply(ReplyType.INFO, self.prompts.get("ask_for_video_url", "请提供视频文件的URL链接。"))
                        e_context['reply'] = reply
                        e_context.action = EventAction.BREAK_PASS
                        return
                    
                    logger.info(f"[ZhipuVision] Processing video URL by {user_id}: {extracted_url}")
                    try:
                        api_messages_payload = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": final_prompt_for_api},
                                    {"type": "video_url", "video_url": {"url": extracted_url, "detail": "auto"}}
                                ]
                            }
                        ]
                        api_response = self._call_glm_api(api_messages_payload)
                        if api_response and "choices" in api_response and api_response["choices"]:
                            reply_text = api_response["choices"][0]["message"]["content"]
                        else:
                            specific_api_message = None
                            if isinstance(api_response, dict):
                                error_info = api_response.get("error")
                                if isinstance(error_info, dict):
                                    specific_api_message = error_info.get("message")
                            
                            if specific_api_message:
                                if not isinstance(specific_api_message, str):
                                    specific_api_message = str(specific_api_message)
                                reply_text = specific_api_message
                            else:
                                reply_text = self.prompts.get("api_error", "抱歉，处理失败")
                        reply = Reply(ReplyType.TEXT, reply_text)
                    except Exception as e:
                        logger.error(f"[ZhipuVision] Error processing video URL: {e}")
                        reply = Reply(ReplyType.TEXT, self.prompts.get("video_process_fail", f"处理失败: {str(e)}"))
                    e_context['reply'] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
        pass

    def _download_media(self, url):
        """下载媒体文件"""
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                return response.read()
        except Exception as e:
            logger.error(f"[ZhipuVision] Failed to download media from {url}: {e}")
            return None

    def _call_glm_api(self, messages_payload):
        api_key = self.api_key
        base_url = self.config.get("api", {}).get("base_url", "https://open.bigmodel.cn/api/paas/v4")
        model = self.config.get("api", {}).get("model", "glm-4v-flash")
        timeout = self.config.get("api", {}).get("timeout", 180)
        temperature = self.config.get("api", {}).get("temperature", 0.8)
        top_p = self.config.get("api", {}).get("top_p", 0.7)
        max_tokens = self.config.get("video", {}).get("max_tokens", 8192)

        headers = {
            "Authorization": f"Bearer {self._generate_token(api_key)}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": messages_payload,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }

        # Create a deep copy of the payload for logging to avoid modifying the original
        log_payload = copy.deepcopy(payload)
        for message in log_payload.get("messages", []):
            if "content" in message and isinstance(message["content"], list):
                for content_item in message["content"]:
                    if isinstance(content_item, dict) and content_item.get("type") == "image_url":
                        image_url_obj = content_item.get("image_url")
                        if isinstance(image_url_obj, dict) and "url" in image_url_obj:
                            url_value = image_url_obj.get("url")
                            if isinstance(url_value, str) and len(url_value) > 250:
                                image_url_obj["url"] = self._truncate_if_long(url_value, max_length=50)
        
        logger.debug(f"[ZhipuVision] Sending request to Zhipu API. Model: {model}. Payload: {json.dumps(log_payload, ensure_ascii=False)}")
        
        try:
            response = requests.post(
                f"{base_url}/chat/completions", 
                headers=headers, 
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.error("[ZhipuVision] Zhipu API request timed out.")
            return {"error": {"code": "timeout", "message": "API request timed out"}}
        except requests.exceptions.HTTPError as http_err:
            truncated_response_text = self._truncate_if_long(response.text, max_length=200)
            logger.error(f"[ZhipuVision] Zhipu API HTTP error: {http_err}. Response: {truncated_response_text}")
            try:
                return response.json()
            except json.JSONDecodeError:
                return {"error": {"code": str(response.status_code), "message": response.text}}
        except Exception as e:
            logger.error(f"[ZhipuVision] Error calling Zhipu API: {e}")
            return {"error": {"code": "unknown", "message": str(e)}}

    def _generate_token(self, apikey: str, exp_seconds=3600):
        try:
            api_key_id, api_key_secret = apikey.split('.')
        except ValueError:
            # This can happen if the apikey is not in the expected "id.secret" format
            raise ValueError("Invalid Zhipu API Key format. Expected 'ID.SECRET'.")

        payload = {
            "api_key": api_key_id,
            "exp": int(round(time.time() * 1000)) + exp_seconds * 1000, # Expiration time in milliseconds
            "timestamp": int(round(time.time() * 1000)), # Current time in milliseconds
        }

        token = jwt.encode(
            payload,
            api_key_secret,
            algorithm="HS256", # Algorithm specified by Zhipu
            headers={"alg": "HS256", "sign_type": "SIGN"} # Headers specified by Zhipu
        )
        return token

    def get_help_text(self, **kwargs):
        kw_image = self.keywords.get('image_analysis', ['z识图'])[0]
        kw_reverse = self.keywords.get('reverse_keyword', ['z反推'])[0]
        kw_i2v = self.keywords.get('i2v_keyword', ['z运镜'])[0]
        kw_vfx = self.keywords.get('vfx_keyword', ['z特效'])[0]
        kw_storyboard = self.keywords.get('storyboard_keyword', ['z分镜'])[0]
        kw_video = self.keywords.get('video_analysis', ['z视频'])[0]
        image_config = self.config.get('image', {})

        base_help = f"""智谱AI视觉（ZhipuVision）插件使用说明：

1. 识图功能:
   - "{kw_image} [图片URL] [可选问题]" (直接使用URL分析)
   - "{kw_image} [你的问题]" (然后发送图片进行分析)
   - 或直接发送 "{kw_image}" (然后发送图片, 使用默认提示)
   - 引用聊天中的图片并回复 "{kw_image} [你的问题]"

2. 反推提示词:
   - "{kw_reverse} [图片URL] [可选风格描述]" (直接使用URL反推)
   - "{kw_reverse} [你的风格描述]" (然后发送图片进行反推)
   - 或直接发送 "{kw_reverse}" (然后发送图片, 使用默认提示)
   - 引用聊天中的图片并回复 "{kw_reverse} [你的风格描述]"

3. 运镜提示词生成:
   - "{kw_i2v} [图片URL] [可选运镜描述]" (直接使用URL生成运镜提示词)
   - "{kw_i2v} [你的运镜描述]" (然后发送图片生成运镜提示词)
   - 或直接发送 "{kw_i2v}" (然后发送图片, 使用默认运镜提示词)
   - 引用聊天中的图片并回复 "{kw_i2v} [你的运镜描述]"

4. 特效提示词生成:
   - "{kw_vfx} [图片URL] [可选特效描述]" (直接使用URL生成特效提示词)
   - "{kw_vfx} [你的特效描述]" (然后发送图片生成特效提示词)
   - 或直接发送 "{kw_vfx}" (然后发送图片, 使用默认特效提示词)
   - 引用聊天中的图片并回复 "{kw_vfx} [你的特效描述]"

5. 分镜脚本生成:
   - "{kw_storyboard} [图片URL] [可选分镜描述]" (直接使用URL生成分镜脚本)
   - "{kw_storyboard} [你的分镜描述]" (然后发送图片生成分镜脚本)
   - 或直接发送 "{kw_storyboard}" (然后发送图片, 使用默认分镜提示词)
   - 引用聊天中的图片并回复 "{kw_storyboard} [你的分镜描述]"

6. 视频分析 (通过URL):
   - "{kw_video} [视频URL] [可选问题]"

通用：
- 在等待图片状态时，可发送 "z取消" 或 "z结束" 来取消当前操作。
- 图片大小限制: {image_config.get('max_size',10)}MB, 视频URL无明确大小限制但受模型处理能力影响。
- 提示：插件配置 (如触发词、部分默认提示) 可在后台的 plugins/ZhipuVision/config.json 文件中修改。
- 运镜、特效和分镜功能使用专业的提示词模板，可从 i2v_prompt.txt、vfx_prompt.txt 和 storyboard_prompt.txt 文件中自定义。
"""
        return base_help
