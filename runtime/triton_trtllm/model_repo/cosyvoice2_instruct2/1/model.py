# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
cosyvoice2_instruct2 Triton Python backend

Implements inference_instruct2 from CosyVoice2:
  - instruct_text  →  LLM prompt text  (controls speaking style/emotion/dialect)
  - reference_wav  →  speaker embedding + speech_feat  (for flow/vocoder voice cloning)
  - llm_prompt_speech_token is intentionally omitted (unlike zero_shot mode)

Pipeline:
  reference_wav ─→ audio_tokenizer  ──────────────────────────────→ prompt_speech_tokens (flow only)
  reference_wav ─→ speaker_embedding ─────────────────────────────→ prompt_spk_embedding (flow only)
  reference_wav ─→ mel_spectrogram   ─────────────────────────────→ prompt_speech_feat   (flow only)
  instruct_text + target_text ────→ tokenizer → LLM  ─────────────→ target_speech_tokens
  prompt_speech_tokens + prompt_speech_feat + prompt_spk_embedding
    + target_speech_tokens  ────────────────────────→ token2wav   ─→ waveform
"""

import json
import os
import threading
import time
from uuid import uuid4

import numpy as np
import torch
from torch.utils.dlpack import from_dlpack, to_dlpack
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer

import torchaudio

from matcha.utils.audio import mel_spectrogram

ORIGINAL_VOCAB_SIZE = 151663
torch.set_num_threads(1)


class TritonPythonModel:
    """Triton Python backend for CosyVoice2 instruct2 mode.

    instruct2 differs from zero_shot in that:
      - instruct_text is used as the LLM text prompt (not reference_text)
      - speech tokens from reference audio are NOT fed to LLM
      - reference audio is still used for speaker embedding and speech features
        (voice timbre cloning via flow/vocoder)
    """

    def initialize(self, args):
        """Initialize model and load required resources.

        Args:
            args: dict with model_config (JSON string)
        """
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args['model_config'])
        parameters = self.model_config['parameters']
        model_params = {k: v["string_value"] for k, v in parameters.items()}
        self.logger.log_info(f"[instruct2] model_params: {model_params}")

        self.dynamic_chunk_strategy = model_params.get(
            "dynamic_chunk_strategy", "exponential"
        )
        self.logger.log_info(
            f"[instruct2] dynamic_chunk_strategy: {self.dynamic_chunk_strategy}"
        )

        # LLM tokenizer
        llm_tokenizer_dir = model_params["llm_tokenizer_dir"]
        self.tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer_dir)

        # instruct2 prompt template:
        # "<|sos|>{instruct_text}{target_text}<|task_id|>"
        # instruct_text supplies the style instruction, target_text is the content.
        self.prompt_template = "<|sos|>{instruct_text}{target_text}<|task_id|>"
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids("<|eos1|>")

        self.device = torch.device("cuda")
        self.decoupled = pb_utils.using_decoupled_model_transaction_policy(
            self.model_config
        )

        # Streaming chunking parameters (same as cosyvoice2)
        self.token_frame_rate = 25
        self.flow_pre_lookahead_len = 3
        self.token_hop_len = 15

        # Default speaker info (fallback when no reference audio is provided)
        spk_info_path = os.path.join(model_params["model_dir"], "spk2info.pt")
        if not os.path.exists(spk_info_path):
            raise ValueError(
                f"spk2info.pt not found in {model_params['model_dir']}"
            )
        spk_info = torch.load(spk_info_path, map_location="cpu", weights_only=False)
        self.default_spk_info = spk_info["001"]
        self.logger.log_info("[instruct2] initialization complete")

    # ------------------------------------------------------------------
    # BLS helper: forward to audio_tokenizer sub-model
    # ------------------------------------------------------------------
#    def forward_audio_tokenizer_bk(self, wav_tensor: pb_utils.Tensor,
#                                wav_len_tensor: pb_utils.Tensor) -> torch.Tensor:
#        """Extract speech tokens from reference audio (used for flow/vocoder only).
#
#        Args:
#            wav_tensor: pb_utils Tensor of reference_wav  (FP32, [1, N])
#            wav_len_tensor: pb_utils Tensor of reference_wav_len (INT32, [1,1])
#
#        Returns:
#            prompt_speech_tokens: torch.Tensor  INT32  [T]
#        """
#        inference_request = pb_utils.InferenceRequest(
#            model_name='audio_tokenizer',
#            requested_output_names=['prompt_speech_tokens'],
#            inputs=[wav_tensor, wav_len_tensor]
#        )
#        inference_response = inference_request.exec()
#        if inference_response.has_error():
#            raise pb_utils.TritonModelException(
#                inference_response.error().message()
#            )
#        prompt_speech_tokens = pb_utils.get_output_tensor_by_name(
#            inference_response, 'prompt_speech_tokens'
#        )
#        return from_dlpack(prompt_speech_tokens.to_dlpack()).cpu()

    def forward_audio_tokenizer(self, wav, wav_len):
        """Forward pass through the audio tokenizer component.

        Args:
            wav: Input waveform tensor
            wav_len: Waveform length tensor

        Returns:
            Tuple of global and semantic tokens
        """
        inference_request = pb_utils.InferenceRequest(
            model_name='audio_tokenizer',
            requested_output_names=['prompt_speech_tokens'],
            inputs=[wav, wav_len]
        )

        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())

        # Extract and convert output tensors
        prompt_speech_tokens = pb_utils.get_output_tensor_by_name(inference_response, 'prompt_speech_tokens')
        prompt_speech_tokens = torch.utils.dlpack.from_dlpack(prompt_speech_tokens.to_dlpack()).cpu()

        return prompt_speech_tokens


    # ------------------------------------------------------------------
    # BLS helper: forward to speaker_embedding sub-model
    # ------------------------------------------------------------------
    def forward_speaker_embedding(self, wav: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding from reference audio.

        Args:
            wav: torch.Tensor FP32 [1, N] at 16kHz

        Returns:
            prompt_spk_embedding: torch.Tensor FP16 [1, D]
        """
        inference_request = pb_utils.InferenceRequest(
            model_name='speaker_embedding',
            requested_output_names=['prompt_spk_embedding'],
            inputs=[pb_utils.Tensor.from_dlpack(
                "reference_wav", to_dlpack(wav)
            )]
        )
        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(
                inference_response.error().message()
            )
        prompt_spk_embedding = pb_utils.get_output_tensor_by_name(
            inference_response, 'prompt_spk_embedding'
        )
        return from_dlpack(prompt_spk_embedding.to_dlpack())

    # ------------------------------------------------------------------
    # BLS helper: forward to tensorrt_llm sub-model
    # ------------------------------------------------------------------
    def forward_llm(self, input_ids: torch.Tensor):
        """Generate speech token IDs from text prompt via LLM.

        Args:
            input_ids: torch.Tensor INT32 [1, L]

        Yields:
            numpy array of generated token IDs (one chunk per yield)
        """
        input_ids_np = input_ids.cpu().numpy()
        max_tokens = 750
        input_dict = {
            "request_output_len":  np.array([[max_tokens]], dtype=np.int32),
            "end_id":              np.array([[self.eos_token_id]], dtype=np.int32),
            "pad_id":              np.array([[self.eos_token_id]], dtype=np.int32),
            "streaming":           np.array([[self.decoupled]], dtype=np.bool_),
            "runtime_top_p":       np.array([[0.95]], dtype=np.float32),
            "runtime_top_k":       np.array([[50]], dtype=np.int32),
            "temperature":         np.array([[0.8]], dtype=np.float32),
            "repetition_penalty":  np.array([[1.1]], dtype=np.float32),
            "random_seed":         np.array([[42]], dtype=np.uint64),
            "input_ids":           input_ids_np,
            "input_lengths":       np.array([[input_ids_np.shape[1]]], dtype=np.int32),
        }
        input_tensor_list = [pb_utils.Tensor(k, v) for k, v in input_dict.items()]

        llm_request = pb_utils.InferenceRequest(
            model_name="tensorrt_llm",
            requested_output_names=["output_ids", "sequence_length"],
            inputs=input_tensor_list,
        )
        llm_responses = llm_request.exec(decoupled=self.decoupled)

        if self.decoupled:
            for llm_response in llm_responses:
                if llm_response.has_error():
                    raise pb_utils.TritonModelException(
                        llm_response.error().message()
                    )
                output_ids = pb_utils.get_output_tensor_by_name(
                    llm_response, "output_ids"
                ).as_numpy()
                seq_lens = pb_utils.get_output_tensor_by_name(
                    llm_response, "sequence_length"
                ).as_numpy()
                yield output_ids[0][0][: seq_lens[0][0]]
        else:
            llm_response = llm_responses
            if llm_response.has_error():
                raise pb_utils.TritonModelException(
                    llm_response.error().message()
                )
            output_ids = pb_utils.get_output_tensor_by_name(
                llm_response, "output_ids"
            ).as_numpy()
            seq_lens = pb_utils.get_output_tensor_by_name(
                llm_response, "sequence_length"
            ).as_numpy()
            yield output_ids[0][0][: seq_lens[0][0]]

    # ------------------------------------------------------------------
    # BLS helper: forward to token2wav sub-model
    # ------------------------------------------------------------------
    def forward_token2wav(
        self,
        target_speech_tokens: torch.Tensor,
        request_id: str,
        prompt_speech_tokens: torch.Tensor = None,
        prompt_speech_feat: torch.Tensor = None,
        prompt_spk_embedding: torch.Tensor = None,
        token_offset: int = None,
        finalize: bool = None,
    ) -> torch.Tensor:
        """Synthesize waveform from speech tokens via flow + vocoder.

        Args:
            target_speech_tokens: generated speech tokens [1, T]
            request_id: unique request ID for stateful streaming
            prompt_speech_tokens: reference speech tokens [1, T'] (flow condition)
            prompt_speech_feat: mel-spectrogram features [1, T'', 80] (flow condition)
            prompt_spk_embedding: speaker embedding [1, D] (flow condition)
            token_offset: streaming chunk start offset
            finalize: whether this is the final chunk

        Returns:
            waveform: torch.Tensor FP32 [1, S]
        """
        target_speech_tokens_tensor = pb_utils.Tensor.from_dlpack(
            "target_speech_tokens", to_dlpack(target_speech_tokens)
        )
        inputs_tensor = [target_speech_tokens_tensor]

        if token_offset is not None:
            assert finalize is not None
            inputs_tensor.append(
                pb_utils.Tensor("token_offset",
                                np.array([[token_offset]], dtype=np.int32))
            )
            inputs_tensor.append(
                pb_utils.Tensor("finalize",
                                np.array([[finalize]], dtype=np.bool_))
            )

        if prompt_spk_embedding is not None:
            assert prompt_speech_feat is not None
            inputs_tensor.extend([
                pb_utils.Tensor.from_dlpack(
                    "prompt_speech_tokens", to_dlpack(prompt_speech_tokens)
                ),
                pb_utils.Tensor.from_dlpack(
                    "prompt_speech_feat", to_dlpack(prompt_speech_feat)
                ),
                pb_utils.Tensor.from_dlpack(
                    "prompt_spk_embedding", to_dlpack(prompt_spk_embedding)
                ),
            ])

        inference_request = pb_utils.InferenceRequest(
            model_name='token2wav',
            requested_output_names=['waveform'],
            inputs=inputs_tensor,
            request_id=request_id,
        )
        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(
                inference_response.error().message()
            )
        waveform = pb_utils.get_output_tensor_by_name(
            inference_response, 'waveform'
        )
        return from_dlpack(waveform.to_dlpack()).cpu()

    # ------------------------------------------------------------------
    # Build LLM input IDs for instruct2:
    #   "<|sos|>{instruct_text}{target_text}<|task_id|>"
    #   NO speech tokens appended (key difference vs zero_shot)
    # ------------------------------------------------------------------
    def parse_input_instruct2(self, target_text: str,
                              instruct_text: str) -> torch.Tensor:
        """Construct LLM input IDs for instruct2 mode.

        Args:
            target_text: the text to synthesize
            instruct_text: speaking style instruction (e.g. '用四川话说')

        Returns:
            input_ids: torch.Tensor INT32 [1, L]
        """
        prompt = self.prompt_template.format(
            instruct_text=instruct_text,
            target_text=target_text
        )
        input_ids = self.tokenizer.encode(prompt)
        return torch.tensor([input_ids], dtype=torch.int32)

    # ------------------------------------------------------------------
    # Mel-spectrogram extraction (same as cosyvoice2 model.py)
    # ------------------------------------------------------------------
    def _extract_speech_feat(self, speech: torch.Tensor) -> torch.Tensor:
        """Compute mel-spectrogram features from 24kHz waveform.

        Args:
            speech: torch.Tensor FP32 [1, N] at 24kHz

        Returns:
            speech_feat: torch.Tensor FP16 [1, T, 80]  (on self.device)
        """
        speech_feat = mel_spectrogram(
            speech,
            n_fft=1920,
            num_mels=80,
            sampling_rate=24000,
            hop_size=480,
            win_size=1920,
            fmin=0,
            fmax=8000
        ).squeeze(dim=0).transpose(0, 1).to(self.device)
        return speech_feat.unsqueeze(dim=0)

    # ------------------------------------------------------------------
    # Background thread: drain LLM token stream
    # ------------------------------------------------------------------
    def _llm_gen_thread(self, generated_ids_iter, semantic_token_ids_arr,
                        llm_is_done_flag):
        for generated_ids in generated_ids_iter:
            generated_ids = generated_ids.tolist()
            if len(generated_ids) == 0:
                break
            semantic_token_ids_arr.extend(generated_ids)
        llm_is_done_flag[0] = True

    # ------------------------------------------------------------------
    # Main execute entry point
    # ------------------------------------------------------------------
    def execute(self, requests):
        """Handle a batch of inference requests.

        Args:
            requests: list of pb_utils.InferenceRequest

        Returns:
            list of pb_utils.InferenceResponse (non-decoupled mode only)
        """
        responses = []

        for request in requests:
            request_id = request.request_id()

            # ── 1. Extract reference audio (optional) ──────────────────
            wav_pb = pb_utils.get_input_tensor_by_name(request, "reference_wav")

            if wav_pb is not None:
                wav_len_pb = pb_utils.get_input_tensor_by_name(
                    request, "reference_wav_len"
                )

                # ── 1a. Audio tokenization (for flow/vocoder only) ──────
                # prompt_speech_tokens are used only in token2wav (flow stage),
                # NOT in LLM stage (key instruct2 distinction)
                prompt_speech_tokens = self.forward_audio_tokenizer(
                    wav_pb, wav_len_pb
                )
                prompt_speech_tokens = prompt_speech_tokens.unsqueeze(0)

                # ── 1b. Prepare 24kHz waveform for mel extraction ───────
                wav_np = wav_pb.as_numpy()
                wav_torch = torch.from_numpy(wav_np)[
                    :, : wav_len_pb.as_numpy()[0][0]
                ]
                wav_24k = torchaudio.transforms.Resample(
                    orig_freq=16000, new_freq=24000
                )(wav_torch)
                speech_feat = self._extract_speech_feat(wav_24k)

                # Align token length with feat length (cosyvoice2 constraint)
                token_len = min(
                    int(speech_feat.shape[1] / 2),
                    prompt_speech_tokens.shape[-1]
                )
                prompt_speech_feat = (
                    speech_feat[:, : 2 * token_len].contiguous().half()
                )
                prompt_speech_tokens = (
                    prompt_speech_tokens[:, :token_len].contiguous()
                )

                # ── 1c. Speaker embedding ───────────────────────────────
                prompt_spk_embedding = self.forward_speaker_embedding(wav_torch)

            else:
                # Fallback: use built-in default speaker
                self.logger.log_info(
                    "[instruct2] no reference_wav provided, using default speaker"
                )
                default = self.default_spk_info
                prompt_speech_tokens = (
                    default["speech_token"] + ORIGINAL_VOCAB_SIZE
                )
                prompt_speech_feat = None
                prompt_spk_embedding = None

            # ── 2. Parse instruct_text and target_text ──────────────────
            instruct_text_pb = pb_utils.get_input_tensor_by_name(
                request, "instruct_text"
            )
            if instruct_text_pb is not None:
                instruct_text = (
                    instruct_text_pb.as_numpy()[0][0].decode("utf-8")
                )
            else:
                instruct_text = ""

            target_text = (
                pb_utils.get_input_tensor_by_name(request, "target_text")
                .as_numpy()[0][0].decode("utf-8")
            )

            self.logger.log_info(
                f"[instruct2] instruct='{instruct_text}' target='{target_text}'"
            )

            # ── 3. Build LLM input (instruct2: no speech tokens in LLM) ─
            input_ids = self.parse_input_instruct2(target_text, instruct_text)

            # ── 4. LLM inference ────────────────────────────────────────
            generated_ids_iter = self.forward_llm(input_ids)
            token2wav_request_id = request_id or str(uuid4())

            # ── 5. Token-to-waveform synthesis ──────────────────────────
            if self.decoupled:
                # Streaming mode: interleave LLM generation and vocoder synthesis
                response_sender = request.get_response_sender()

                semantic_token_ids_arr = []
                llm_is_done_flag = [False]

                llm_thread = threading.Thread(
                    target=self._llm_gen_thread,
                    args=(generated_ids_iter, semantic_token_ids_arr,
                          llm_is_done_flag)
                )
                llm_thread.start()

                token_offset = 0
                chunk_index = 0
                start_time = time.time()
                this_token_hop_len = self.token_hop_len

                while True:
                    pending_num = (
                        len(semantic_token_ids_arr) - token_offset
                    )

                    if llm_is_done_flag[0]:
                        break

                    if (pending_num
                            >= this_token_hop_len + self.flow_pre_lookahead_len):
                        this_tts_speech_token = semantic_token_ids_arr[
                            : token_offset + this_token_hop_len
                              + self.flow_pre_lookahead_len
                        ]
                        this_tts_speech_token = (
                            torch.tensor(this_tts_speech_token)
                            .unsqueeze(dim=0)
                            .to(torch.int32)
                            .to(self.device)
                        )

                        sub_tts_speech = self.forward_token2wav(
                            this_tts_speech_token,
                            token2wav_request_id,
                            prompt_speech_tokens,
                            prompt_speech_feat,
                            prompt_spk_embedding,
                            token_offset,
                            False,
                        )

                        audio_tensor = pb_utils.Tensor.from_dlpack(
                            "waveform", to_dlpack(sub_tts_speech)
                        )
                        response_sender.send(
                            pb_utils.InferenceResponse(
                                output_tensors=[audio_tensor]
                            )
                        )

                        token_offset += this_token_hop_len
                        self.logger.log_info(
                            f"[instruct2] chunk={chunk_index}, "
                            f"hop_len={this_token_hop_len}"
                        )

                        if self.dynamic_chunk_strategy == "exponential":
                            this_token_hop_len = (
                                self.token_frame_rate * (2 ** chunk_index)
                            )
                        elif self.dynamic_chunk_strategy == "time_based":
                            cost_time = time.time() - start_time
                            duration = token_offset / self.token_frame_rate
                            if chunk_index > 0 and cost_time > 0:
                                avg_chunk_time = cost_time / (chunk_index + 1)
                                if avg_chunk_time > 0:
                                    multiples = (
                                        (duration - cost_time) / avg_chunk_time
                                    )
                                    self.logger.log_info(
                                        f"[instruct2] multiples={multiples:.2f}"
                                    )
                                    next_pending = (
                                        len(semantic_token_ids_arr)
                                        - token_offset
                                    )
                                    if multiples > 4:
                                        this_token_hop_len = (
                                            (next_pending
                                             // self.token_hop_len + 1)
                                            * self.token_hop_len
                                        )
                                    elif multiples > 2:
                                        this_token_hop_len = (
                                            (next_pending
                                             // self.token_hop_len)
                                            * self.token_hop_len
                                        )
                                    else:
                                        this_token_hop_len = self.token_hop_len
                                    this_token_hop_len = max(
                                        self.token_hop_len, this_token_hop_len
                                    )
                        chunk_index += 1
                    else:
                        time.sleep(0.02)

                # Final chunk with all remaining tokens
                this_tts_speech_token = (
                    torch.tensor(semantic_token_ids_arr)
                    .unsqueeze(dim=0)
                    .to(torch.int32)
                    .to(self.device)
                )
                sub_tts_speech = self.forward_token2wav(
                    this_tts_speech_token,
                    token2wav_request_id,
                    prompt_speech_tokens,
                    prompt_speech_feat,
                    prompt_spk_embedding,
                    token_offset,
                    True,
                )
                audio_tensor = pb_utils.Tensor.from_dlpack(
                    "waveform", to_dlpack(sub_tts_speech)
                )
                response_sender.send(
                    pb_utils.InferenceResponse(output_tensors=[audio_tensor])
                )

                llm_thread.join()
                response_sender.send(
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )
                self.logger.log_info(
                    "[instruct2] sent TRITONSERVER_RESPONSE_COMPLETE_FINAL"
                )

            else:
                # Offline (non-streaming) mode
                generated_ids = next(generated_ids_iter)
                generated_ids = (
                    torch.tensor(generated_ids)
                    .unsqueeze(0)
                    .to(self.device)
                )
                if generated_ids is None or len(generated_ids) == 0:
                    raise pb_utils.TritonModelException(
                        "[instruct2] Generated IDs is None or empty"
                    )

                audio = self.forward_token2wav(
                    generated_ids,
                    token2wav_request_id,
                    prompt_speech_tokens,
                    prompt_speech_feat,
                    prompt_spk_embedding,
                )

                audio_tensor = pb_utils.Tensor.from_dlpack(
                    "waveform", to_dlpack(audio)
                )
                responses.append(
                    pb_utils.InferenceResponse(output_tensors=[audio_tensor])
                )

        if not self.decoupled:
            return responses
