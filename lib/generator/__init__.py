from lib.generator.gfn import FMGFlowNetGenerator, TBGFlowNetGenerator


def get_generator(tokenizer, reward_exp_min, task='tfbind', vocab_size=4, max_len=8, device='cpu',
                  gen_do_explicit_Z=0, gen_learning_rate=5e-4, gen_Z_learning_rate=5e-3,
                  ):
    if not gen_do_explicit_Z:
        return FMGFlowNetGenerator(tokenizer,
                                   device=device, task=task, vocab_size=vocab_size, max_len=max_len,
                                   gen_do_explicit_Z=gen_do_explicit_Z, gen_learning_rate=gen_learning_rate,)
    else:
        return TBGFlowNetGenerator(tokenizer, reward_exp_min,
                                   device=device, task=task, vocab_size=vocab_size, max_len=max_len,
                                   gen_do_explicit_Z=gen_do_explicit_Z, gen_learning_rate=gen_learning_rate,
                                   gen_Z_learning_rate=gen_Z_learning_rate,)
