from dcformer import DCFormerLlama

def match_weight(model):
    map_dict={'q_proj':'query', 'k_proj':'key', 'v_proj':'value','o_proj':'post', 'gate_proj': 'ffn_layer1_gate', 'up_proj': 'ffn_layer1', 'down_proj': 'ffn_layer2',
              'weight': 'w'} # 'pre_proj': 'pre_proj', 'post_proj': 'post_proj'
    L, E, H, D = w['state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.self_attention.key.w'].shape
    N = w['state.mdl_vars.params.lm.embedding_lookup.emb_var'].shape[0]
    state_dict = {}
    for k, v in model.named_parameters():
        if k == 'model.embed_tokens.weight':
            v = w['state.mdl_vars.params.lm.embedding_lookup.emb_var'][:50257,:]
        elif k == 'model.norm.weight':
            v = w['state.mdl_vars.params.lm.final_ln.scale']
        elif k == 'lm_head.weight':
            v = w['state.mdl_vars.params.lm.softmax.logits_ffn.linear.w'].T  # E,N -> N,E
            #v = torch.zeros_like(v)
            #_v = w['state.mdl_vars.params.lm.softmax.logits_ffn.linear.w'].T  # E,N -> N,E
            #v[:_v.shape[0],:] = torch.tensor(_v) # pad unembedding matrix as 0
        else:
            layer = int(k.split('.')[2])
            if 'self_attn' in k:
                if k.endswith('_m'):continue # merged proj weights
                _, _, _, _, ptype, wtype = k.split('.')
                if k.endswith('_p'): continue # ablation parameters
                if ptype in ['pre_proj', 'post_proj']: # pre post proj
                    v = w[f'state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.self_attention.{map_dict.get(ptype, ptype)}.{map_dict.get(wtype, wtype)}'][layer]
                else: # qkov
                    v = w[f'state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.self_attention.{map_dict.get(ptype, ptype)}.{map_dict.get(wtype, wtype)}'][layer].reshape(E,H*D).T
                    if ptype == 'o_proj': v = v.T
                #print(ptype, wtype, v.max(), v.min(), v.var())
            elif 'mlp' in k:
                ptype = k.split('.')[4] # gate, up, down proj
                v = w[f'state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.ff_layer.{map_dict[ptype]}.linear.w'][layer].T
            elif 'post_attention_layernorm' in k: # mlp layernorm
                v = w['state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.ff_layer.layer_norm.scale'][layer]
            elif 'input_layernorm' in k: # attention layernorm
                v = w['state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.layer_norm.scale'][layer]
        state_dict[k] = torch.tensor(v)
    model.load_state_dict(state_dict, strict=False)
    return model 

model_size_str = '2p8B'
window_size=256
kwargs=dict(window_size=window_size)
model = DCFormerLlama.from_name(model_size_str, **kwargs)
checkpoint_path = '/home/mengqy/Data/models/PileDCLlama3B2Kx4x256x1DWDDLR00032v4_checkpoint_00143000.torch.bin'
w = torch.load(checkpoint_path)
model = match_weight(model,w)
