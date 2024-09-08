from .sampler import GPTSampler


def build_gpt_sampler(args, num_vq_embeds=1024):
    if args.sampler == 'gpt_large':
        gpt_config = {
            'num_layers': 32,
            'num_heads': 25,
            'embed_dim': 1600,
            'max_seq_len': 512,
            'rope_theta': 50000,
            'sos_token_id': 0,
            }
    elif args.sampler ==  'gpt_medium':
        gpt_config = {
            'num_layers': 24,
            'num_heads': 16,
            'embed_dim': 1024,
            'max_seq_len': 512,
            'rope_theta': 50000,
            'sos_token_id': 0,
            }
    elif args.sampler ==  'gpt_base':
        gpt_config = {
            'num_layers': 12,
            'num_heads': 12,
            'embed_dim': 768,
            'max_seq_len': 512,
            'rope_theta': 50000,
            'sos_token_id': 0,
            }
    elif args.sampler ==  'gpt_small':
        gpt_config = {
            'num_layers': 10,
            'num_heads': 8,
            'embed_dim': 512,
            'max_seq_len': 512,
            'rope_theta': 50000,
            'sos_token_id': 0,
            }
    else:
        raise NotImplementedError(f"Unknown scale for VQGANSampler: {args.sampler}")

    print(f" - GPT sampler version : {args.sampler}")
    gpt_config["vocab_size"] = num_vq_embeds
    
    return GPTSampler(gpt_config)
