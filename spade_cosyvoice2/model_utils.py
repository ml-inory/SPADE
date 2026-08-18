"""Shared model/data helpers for SPADE-on-CosyVoice2."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch.utils.data import DataLoader

from spade_cosyvoice2.paths import ensure_import
from spade_cosyvoice2.prune_llm import shrink_qwen2_for_llm


def load_configs(model_dir: str):
    """Load the CosyVoice2 hyperpyyaml config like CosyVoice2.__init__."""
    ensure_import()
    from hyperpyyaml import load_hyperpyyaml

    with open(f"{model_dir}/cosyvoice2.yaml", "r") as fh:
        return load_hyperpyyaml(
            fh, overrides={"qwen_pretrain_path": f"{model_dir}/CosyVoice-BlankEN"}
        )


def build_llm(configs, device: str | torch.device, llm_pt: str | None = None,
              retained: list[int] | None = None):
    """Construct an independent Qwen2LM and load a (possibly pruned) llm.pt.

    The configs' ``llm`` instance is deep-copied per call so that the teacher
    and the student never share modules. The BlankEN-pretrained Qwen2 inside
    the copy is replaced by a blank model of the same architecture, since
    ``llm.pt`` overwrites it anyway.
    """
    from transformers import AutoModelForCausalLM

    src = configs["llm"]
    extractor = getattr(src, "speech_token_extractor", None)
    src.speech_token_extractor = None  # onnx session is not deepcopy-able
    try:
        llm = deepcopy(src)
    finally:
        src.speech_token_extractor = extractor
    qwen_config = deepcopy(llm.llm.model.config)
    llm.llm.model = AutoModelForCausalLM.from_config(qwen_config)
    if retained is not None:
        shrink_qwen2_for_llm(llm, retained)
    if llm_pt is not None:
        state_dict = torch.load(llm_pt, map_location=device, weights_only=True)
        llm.load_state_dict(state_dict, strict=retained is None)
    return llm.to(device).eval()


def build_dataloader(configs, data_list: str, num_workers: int = 0) -> DataLoader:
    """CosyVoice's dynamic-batch dataset pipeline wrapped in a DataLoader."""
    from cosyvoice.dataset.dataset import Dataset

    dataset = Dataset(
        data_list,
        data_pipeline=configs["data_pipeline"],
        mode="train",
        shuffle=True,
        partition=False,
    )
    return DataLoader(dataset, batch_size=None, num_workers=num_workers)


def spade_forward(llm, batch: dict, device: str | torch.device) -> dict:
    """Replicate Qwen2LM.forward input construction, exposing hidden states,
    attention maps, and the embedding output for SPADE distillation.
    """
    from cosyvoice.utils.mask import make_pad_mask

    text_token = batch["text_token"].to(device)
    text_token_len = batch["text_token_len"].to(device)
    if "speech_token" not in batch:  # pragma: no cover - offline tokens expected
        raise ValueError("distillation requires offline speech_token in the batch")
    speech_token = batch["speech_token"].to(device)
    speech_token_len = batch["speech_token_len"].to(device)

    text_token_emb = llm.llm.model.model.embed_tokens(text_token)
    speech_token_emb = llm.speech_embedding(speech_token)
    sos_emb = llm.llm_embedding.weight[llm.sos].reshape(1, 1, -1)
    task_id_emb = llm.llm_embedding.weight[llm.task_id].reshape(1, 1, -1)

    lm_target, lm_input, lm_input_len = llm.prepare_lm_input_target(
        sos_emb,
        text_token,
        text_token_emb,
        text_token_len,
        task_id_emb,
        speech_token,
        speech_token_emb,
        speech_token_len,
    )
    masks = ~make_pad_mask(lm_input_len.to(device), lm_input.size(1))
    outs = llm.llm.model(
        inputs_embeds=lm_input.to(device),
        attention_mask=masks,
        output_hidden_states=True,
        output_attentions=True,
    )
    # HF includes the embedding output at index 0; strip it so the list
    # follows SPADE's convention: [post-block 0 .. n-1].
    hidden_states = list(outs.hidden_states)[1:]
    lm_target = lm_target.to(device)
    return {
        "logits": llm.llm_decoder(outs.hidden_states[-1]),
        "hidden_states": hidden_states,
        "attentions": list(outs.attentions),
        "embedding_out": outs.hidden_states[0],
        "lm_target": lm_target,
        "lm_target_masked": torch.where(
            lm_target == -1, torch.tensor(-100, device=device), lm_target
        ),
    }


def load_cosyvoice2_with_llm(
    model_dir: str,
    llm_pt: str | None = None,
    retained: list[int] | None = None,
    fp16: bool = False,
):
    """Build a CosyVoice2 object whose LLM is (optionally) pruned."""
    ensure_import()
    from cosyvoice.cli.cosyvoice import CosyVoice2

    cosyvoice = CosyVoice2(model_dir, fp16=fp16)
    if retained is not None:
        if llm_pt is None:
            raise ValueError("llm_pt is required when retained is set")
        shrink_qwen2_for_llm(cosyvoice.model.llm, retained)
        state_dict = torch.load(llm_pt, map_location=cosyvoice.model.device, weights_only=True)
        cosyvoice.model.llm.load_state_dict(state_dict, strict=False)
        cosyvoice.model.llm.to(cosyvoice.model.device).eval()
    return cosyvoice


@torch.no_grad()
def synthesize(
    cosyvoice,
    text: str,
    prompt_text: str,
    prompt_wav: str,
    text_frontend: bool = False,
) -> torch.Tensor:
    """Zero-shot synthesis of ``text`` given a prompt; returns the waveform."""
    model_input = cosyvoice.frontend.frontend_zero_shot(
        text, prompt_text, prompt_wav, cosyvoice.sample_rate, ""
    )
    for output in cosyvoice.model.tts(**model_input, stream=False):
        return output["tts_speech"]
    raise RuntimeError("no speech generated")
