"""
Merge a PEFT LoRA adapter into a Qwen2.5-VL checkpoint (W <- W + scale * B @ A) and save a full model.

Done directly on the safetensors shards, so it does not depend on the peft version the adapter was
trained with, and it keeps the base checkpoint's extra tokens (e.g. <POS_INDICATOR>/<POS_EMBEDDING>).
Handles the old (model.layers / visual.blocks) and new (model.language_model.layers / model.visual.blocks)
transformers key layouts.

Example:
  python tools/merge_lora.py \
      --base ckpts/Qwen2.5-VL-7B-Instruct-with-new-special-tokens \
      --lora Qwen2.5-VL-7B_Trained_v1.3_llm_lora/models/DepthLM \
      --out  ckpts/Qwen2.5-VL-7B-Instruct-with-new-special-tokens-DepthLM
"""
import argparse, json, os, shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def candidate_keys(peft_module):
    # 'base_model.model.<module>' -> possible base checkpoint keys for '<module>.weight'
    m = peft_module[len('base_model.model.'):] if peft_module.startswith('base_model.model.') else peft_module
    cands = [m]
    if m.startswith('model.layers.'):
        cands.append('model.language_model.' + m[len('model.'):])
    if m.startswith('visual.'):
        cands.append('model.' + m)
    if m.startswith('model.language_model.'):
        cands.append('model.' + m[len('model.language_model.'):])
    if m.startswith('model.visual.'):
        cands.append(m[len('model.'):])
    return [c + '.weight' for c in cands]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--lora', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.lora, 'adapter_config.json')))
    assert cfg['peft_type'] == 'LORA' and not cfg.get('use_dora') and not cfg.get('rank_pattern') \
        and not cfg.get('alpha_pattern') and not cfg.get('modules_to_save'), 'unsupported adapter config'
    r, alpha = cfg['r'], cfg['lora_alpha']
    scale = alpha / (r ** 0.5) if cfg.get('use_rslora') else alpha / r
    print(f'LoRA r={r} alpha={alpha} scale={scale}')

    index = json.load(open(os.path.join(args.base, 'model.safetensors.index.json')))
    weight_map = index['weight_map']

    # collect (A, B) per target base key
    lora = {}
    with safe_open(os.path.join(args.lora, 'adapter_model.safetensors'), 'pt') as f:
        modules = sorted({k.rsplit('.lora_', 1)[0] for k in f.keys()})
        for mod in modules:
            target = next((c for c in candidate_keys(mod) if c in weight_map), None)
            if target is None:
                raise KeyError(f'no base weight for LoRA module {mod}')
            lora[target] = (f.get_tensor(mod + '.lora_A.weight'), f.get_tensor(mod + '.lora_B.weight'))
    print(f'{len(lora)} LoRA modules mapped onto base weights')

    os.makedirs(args.out, exist_ok=True)
    for name in os.listdir(args.base):  # configs, tokenizer, processor, index
        src = os.path.join(args.base, name)
        if os.path.isfile(src) and not name.endswith('.safetensors'):
            shutil.copy2(src, os.path.join(args.out, name))

    merged = 0
    for shard in sorted(set(weight_map.values())):
        tensors = {}
        with safe_open(os.path.join(args.base, shard), 'pt') as f:
            metadata = f.metadata()
            for k in f.keys():
                w = f.get_tensor(k)
                if k in lora:
                    A, B = lora[k]
                    delta = (B.float() @ A.float()) * scale
                    assert delta.shape == w.shape, (k, delta.shape, w.shape)
                    w = (w.float() + delta).to(w.dtype)
                    merged += 1
                tensors[k] = w.contiguous()
        save_file(tensors, os.path.join(args.out, shard), metadata=metadata)
        print(f'wrote {shard}')
    assert merged == len(lora), f'merged {merged} of {len(lora)} modules'
    print(f'done: merged {merged} modules -> {args.out}')


if __name__ == '__main__':
    main()
